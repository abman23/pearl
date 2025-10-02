import json
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaModel

from preprocess.utils import index_to_parameter


class PEARL_MPS(nn.Module):
    def __init__(
        self,
        device,
        model_name: str = "unsloth/Llama-3.2-3B",
        min_thr: int = -80,
        max_thr: int = -60,
        rank: int = 128,
        with_head: bool = False,
        with_adapter: bool = False,
        default_task: str = "thr",
        wa_head_type: str = "linear",  # "linear", "2layer", "3layer"
    ):
        super().__init__()

        if with_head:
            # print(f"Initializing heads...")
            # Load the base Llama model
            # self.lm = LlamaModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            lm_config = AutoConfig.from_pretrained(model_name)
            # print(self.lm.config.name_or_path)
            hidden_size = lm_config.hidden_size
            # Task-specific heads
            self.thr_head = initialize_classification_head(
                nn.Linear(hidden_size, max_thr - min_thr, dtype=torch.float16, device=device)
            )
            self.bssid_head = initialize_classification_head(
                nn.Linear(hidden_size, 10, dtype=torch.float16, device=device)
            )
            self.wa_head = initialize_classification_head(
                nn.Linear(hidden_size, len(index_to_parameter), dtype=torch.float16, device=device)
            )
            self.wa_head_2layer = nn.Sequential(
                nn.Linear(hidden_size, 256, dtype=torch.float16, device=device),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(index_to_parameter), dtype=torch.float16, device=device),
            )
            self.wa_head_3layer = nn.Sequential(
                nn.Linear(hidden_size, 256, dtype=torch.float16, device=device),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64, dtype=torch.float16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, len(index_to_parameter), dtype=torch.float16, device=device),
            )

        # Clear GPU memory before loading to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # print(f"Loading base model {model_name}... (this may take a few minutes for large models)")
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,  # Reduce memory usage during loading
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        if with_adapter:
            # Apply PEFT by adding LoRA adapters
            peft_config = LoraConfig(
                # task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=rank,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            # print(f"Initializing LoRA adapters...")
            self.lm = get_peft_model(self.lm, peft_config, adapter_name="thr")
            self.lm.add_adapter("bssid", peft_config)
            self.lm.add_adapter("wa", peft_config)
            # print(f"LoRA adapters initialized successfully!")
        else:
            # freeze the model parameters when not using adapter
            for param in self.lm.parameters():
                param.requires_grad = False

        self.min_thr = min_thr
        self.max_thr = max_thr
        self.rank = rank
        self.with_head = with_head
        self.with_adapter = with_adapter
        self.default_task = default_task
        self.wa_head_type = wa_head_type
        self.base_model_name = model_name

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> torch.Tensor:
        if task is None:
            task = self.default_task

        if self.with_adapter:
            if task == "thr":
                self.lm.set_adapter("thr")
            elif task == "bssid":
                self.lm.set_adapter("bssid")
            elif task == "wa":
                self.lm.set_adapter("wa")

        if self.with_head:
            outputs = self.lm(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
            )

            # Get the last hidden state
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            # hidden_state_mean = last_hidden_state.mean(dim=1)

            # Apply classification head
            if task == "thr":
                outputs = self.thr_head(hidden_state)  # Take the last token's hidden state
            elif task == "bssid":
                outputs = self.bssid_head(hidden_state)  # Take the last token's hidden state
            elif task == "wa":
                if self.wa_head_type == "linear":
                    outputs = self.wa_head(hidden_state)  # Take the last token's hidden state
                elif self.wa_head_type == "2layer":
                    outputs = self.wa_head_2layer(hidden_state)  # Take the last token's hidden state
                elif self.wa_head_type == "3layer":
                    outputs = self.wa_head_3layer(hidden_state)  # Take the last token's hidden state
        else:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs

    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: str = "wa",
        sampling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the predicted value for the input."""
        if self.with_head:
            logits = self(input_ids, attention_mask, task=task)
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
        else:
            if self.with_adapter:
                if task == "thr":
                    self.lm.set_adapter("thr")
                elif task == "bssid":
                    self.lm.set_adapter("bssid")
                elif task == "wa":
                    self.lm.set_adapter("wa")

            generated_ids = self.lm.generate(input_ids, max_new_tokens=128)
            # predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            new_tokens = generated_ids[:, input_ids.shape[1] :]
            predictions = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            logits = None
        return predictions, logits

    def save(self, path: str, merge_adapter: bool = False):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.with_adapter:
            if merge_adapter:
                self.lm.set_adapter("wa")
                self.lm.delete_adapter("thr")
                self.lm.delete_adapter("bssid")
                self.lm = self.lm.merge_and_unload()
            self.lm.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        if self.with_head:
            torch.save(self.thr_head.state_dict(), os.path.join(path, "thr_head.pt"))
            torch.save(self.bssid_head.state_dict(), os.path.join(path, "bssid_head.pt"))
            torch.save(self.wa_head.state_dict(), os.path.join(path, "wa_head.pt"))
            torch.save(self.wa_head_2layer.state_dict(), os.path.join(path, "wa_head_2layer.pt"))
            torch.save(self.wa_head_3layer.state_dict(), os.path.join(path, "wa_head_3layer.pt"))
        config = {
            "base_model_name": self.base_model_name,
            "min_thr": self.min_thr,
            "max_thr": self.max_thr,
            "rank": self.rank,
            "with_head": self.with_head,
            "with_adapter": self.with_adapter,
            "default_task": self.default_task,
            "wa_head_type": self.wa_head_type,
            "merge_adapter": merge_adapter,
        }
        json.dump(config, open(os.path.join(path, "pearl_config.json"), "w"))

    @classmethod
    def from_pretrained(cls, path: str, device="cuda", **kwargs) -> "PEARL_MPS":
        start_time = time.time()
        # Check if path is a local directory
        if os.path.exists(path) and os.path.isdir(path):
            # Load from local path
            config = json.load(open(os.path.join(path, "pearl_config.json"), "r"))

            # Initialize model without LoRA first
            model = cls(
                device=device,
                model_name=config["base_model_name"],
                min_thr=config["min_thr"],
                max_thr=config["max_thr"],
                rank=config["rank"],
                with_head=config["with_head"],
                with_adapter=config["with_adapter"],
                default_task=config["default_task"],
                wa_head_type=config["wa_head_type"],
            )

            if model.with_adapter:
                print(f"Loading LoRA adapters...")

                # Check adapter file sizes
                for adapter_name in ["thr", "bssid", "wa"]:
                    adapter_path = os.path.join(path, adapter_name)
                    if os.path.exists(adapter_path):
                        total_size = 0
                        for root, dirs, files in os.walk(adapter_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                total_size += os.path.getsize(file_path)
                        print(f"  {adapter_name} adapter size: {total_size / 1024**2:.1f} MB")

                if not kwargs.get("load_wa_only", True):
                    # print(f"  Loading 'thr' adapter...")
                    model.lm.load_adapter(os.path.join(path, "thr"), adapter_name="thr")
                    # print(f"  Loading 'bssid' adapter...")
                    model.lm.load_adapter(os.path.join(path, "bssid"), adapter_name="bssid")
                print(f"  Loading 'wa' adapter...")
                model.lm.load_adapter(os.path.join(path, "wa"), adapter_name="wa")
                print(f"LoRA adapters loaded successfully!")

            if model.with_head:
                print(f"Loading heads...")

                # Check head file sizes
                for head_name in [
                    "thr_head.pt",
                    "bssid_head.pt",
                    "wa_head.pt",
                    "wa_head_2layer.pt",
                    "wa_head_3layer.pt",
                ]:
                    head_path = os.path.join(path, head_name)
                    if os.path.exists(head_path):
                        size_mb = os.path.getsize(head_path) / 1024**2
                        print(f"  {head_name} size: {size_mb:.1f} MB")

                # Load state dicts directly to device to avoid slow CPU-to-GPU transfer
                if not kwargs.get("load_wa_only", True):
                    model.thr_head.load_state_dict(torch.load(os.path.join(path, "thr_head.pt"), map_location=device))
                    model.bssid_head.load_state_dict(
                        torch.load(os.path.join(path, "bssid_head.pt"), map_location=device)
                    )
                model.wa_head.load_state_dict(torch.load(os.path.join(path, "wa_head.pt"), map_location=device))
                model.wa_head_2layer.load_state_dict(
                    torch.load(os.path.join(path, "wa_head_2layer.pt"), map_location=device)
                )
                model.wa_head_3layer.load_state_dict(
                    torch.load(os.path.join(path, "wa_head_3layer_fp16.pt"), map_location=device)
                )
                print(f"Heads loaded successfully!")
        else:
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
