import json
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

from preprocess.utils import index_to_parameter


class MiniLM(nn.Module):
    def __init__(
        self,
        device,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_thr: int = -80,
        max_thr: int = -60,
        rank: int = 128,
        with_head: bool = False,
        with_adapter: bool = False,
        default_task: str = "thr",
        wa_head_type: str = "linear",  # "linear", "2layer", "3layer"
    ):
        super().__init__()

        hidden_size = 384
        # Task-specific heads
        self.thr_head = initialize_classification_head(
            nn.Linear(hidden_size, max_thr - min_thr, dtype=torch.float32, device=device)
        )
        self.bssid_head = initialize_classification_head(nn.Linear(hidden_size, 10, dtype=torch.float32, device=device))
        self.wa_head = initialize_classification_head(
            nn.Linear(hidden_size, len(index_to_parameter), dtype=torch.float32, device=device)
        )
        self.wa_head_2layer = nn.Sequential(
            nn.Linear(hidden_size, 256, dtype=torch.float32, device=device),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(index_to_parameter), dtype=torch.float32, device=device),
        )
        self.wa_head_3layer = nn.Sequential(
            nn.Linear(hidden_size, 256, dtype=torch.float32, device=device),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(index_to_parameter), dtype=torch.float32, device=device),
        )

        # Clear GPU memory before loading to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # print(f"Loading base model {model_name}... (this may take a few minutes for large models)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype="auto")

        if with_adapter:
            # Apply PEFT by adding LoRA adapters
            peft_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(self.model, peft_config, adapter_name="wa")
        else:
            # freeze the model parameters when not using adapter
            for param in self.model.parameters():
                param.requires_grad = False

        self.min_thr = min_thr
        self.max_thr = max_thr
        self.rank = rank
        self.with_head = with_head
        self.with_adapter = with_adapter
        self.default_task = default_task
        self.wa_head_type = wa_head_type
        self.model_name = model_name

    def forward(
        self,
        input,
        task: Optional[str] = None,
    ) -> torch.Tensor:
        if task is None:
            task = self.default_task

        base_outputs = self.model(**input)
        sentence_embeddings = mean_pooling(base_outputs, input.get("attention_mask"))

        # Apply classification head
        if task == "thr":
            outputs = self.thr_head(sentence_embeddings)
        elif task == "bssid":
            outputs = self.bssid_head(sentence_embeddings)
        elif task == "wa":
            if self.wa_head_type == "linear":
                outputs = self.wa_head(sentence_embeddings)
            elif self.wa_head_type == "2layer":
                outputs = self.wa_head_2layer(sentence_embeddings)
            elif self.wa_head_type == "3layer":
                outputs = self.wa_head_3layer(sentence_embeddings)
        return outputs

    def predict(self, input: dict, task: str = "thr", sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the predicted value for the input."""

        logits = self(input, task=task)
        if sampling:
            probs = torch.softmax(logits, dim=-1)

            p = 0.9  # top-p sampling
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask out tokens with cumulative probability > p
            cutoff_mask = cumulative_probs > p

            # Set masked-out logits to a large negative number (log(0))
            sorted_probs[cutoff_mask] = 0.0

            # Re-normalize after masking
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sample from the filtered distribution
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)

            # Map back to original indices
            predictions = sorted_indices.gather(-1, sampled_idx).squeeze(-1)
            # predictions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            predictions = torch.argmax(logits, dim=-1)

        return predictions, logits

    def save(self, path: str, merge_adapter: bool = False):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.with_adapter:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        torch.save(self.thr_head.state_dict(), os.path.join(path, "thr_head.pt"))
        torch.save(self.bssid_head.state_dict(), os.path.join(path, "bssid_head.pt"))
        torch.save(self.wa_head.state_dict(), os.path.join(path, "wa_head.pt"))
        torch.save(self.wa_head_2layer.state_dict(), os.path.join(path, "wa_head_2layer.pt"))
        torch.save(self.wa_head_3layer.state_dict(), os.path.join(path, "wa_head_3layer.pt"))
        config = {
            "min_thr": self.min_thr,
            "max_thr": self.max_thr,
            "rank": self.rank,
            "with_head": self.with_head,
            "with_adapter": self.with_adapter,
            "default_task": self.default_task,
            "wa_head_type": self.wa_head_type,
            "model_name": self.model_name,
        }
        json.dump(config, open(os.path.join(path, "miniLM_config.json"), "w"))

    @classmethod
    def from_pretrained(cls, path: str, device="cuda", **kwargs) -> "MiniLM":
        start_time = time.time()
        # Check if path is a local directory
        if os.path.exists(path) and os.path.isdir(path):
            # Load from local path
            config = json.load(open(os.path.join(path, "miniLM_config.json"), "r"))
            model = cls(
                device=device,
                model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                min_thr=config["min_thr"],
                max_thr=config["max_thr"],
                rank=config["rank"],
                with_head=config["with_head"],
                with_adapter=config["with_adapter"],
                default_task=config["default_task"],
                wa_head_type=config["wa_head_type"],
            )
            print(f"Loading heads...")

            # Check head file sizes
            for head_name in ["thr_head.pt", "bssid_head.pt", "wa_head.pt", "wa_head_2layer.pt", "wa_head_3layer.pt"]:
                head_path = os.path.join(path, head_name)
                if os.path.exists(head_path):
                    size_mb = os.path.getsize(head_path) / 1024**2
                    print(f"  {head_name} size: {size_mb:.1f} MB")

            # Load state dicts directly to device to avoid slow CPU-to-GPU transfer
            if not kwargs.get("load_wa_only", True):
                model.thr_head.load_state_dict(torch.load(os.path.join(path, "thr_head.pt"), map_location=device))
                model.bssid_head.load_state_dict(torch.load(os.path.join(path, "bssid_head.pt"), map_location=device))
            model.wa_head.load_state_dict(torch.load(os.path.join(path, "wa_head.pt"), map_location=device))
            model.wa_head_2layer.load_state_dict(
                torch.load(os.path.join(path, "wa_head_2layer.pt"), map_location=device)
            )
            model.wa_head_3layer.load_state_dict(
                torch.load(os.path.join(path, "wa_head_3layer.pt"), map_location=device)
            )
            print(f"Heads loaded successfully!")

            if model.with_adapter:
                model.model.load_adapter(os.path.join(path, "wa"), adapter_name="wa")
                model.model.set_adapter("wa")
        else:
            # Load from model ID
            kwargs.pop("load_wa_only", None)
            model = cls(device=device, model_name=path, **kwargs)
        print_trainable_parameters(model)

        end_time = time.time()
        print(f"Model loaded in {end_time - start_time: .2f} seconds")

        return model


def print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def initialize_classification_head(head):
    """Initialize classification head for better training stability."""

    # no special initialization based on test results
    # Xavier uniform for classification
    # init.xavier_uniform_(head.weight, gain=0.5)
    # init.zeros_(head.bias)

    return head


# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
