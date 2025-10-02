"""Supervised fine-tuning for MiniLM model on Wi-Fi Aware parameter tuning.

"""
import argparse
import math
import os.path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from model.MiniLM import MiniLM
from preprocess.wa_dataset import WADataset


def collate_fn(model):
    def _collate_fn(batch):
        # Preprocess the prompts and labels
        prompts = [example["prompt"] for example in batch]
        rewards_list = [example["rewards"] for example in batch]
        scores = [[rewards[key] for key in sorted(rewards.keys())] for rewards in rewards_list]
        target_probs = torch.softmax(torch.tensor(scores, dtype=torch.bfloat16), dim=-1)

        tokenized = model.tokenizer(text=prompts, padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor([example["opt_parameters"] for example in batch])

        return (tokenized, labels, target_probs)

    return _collate_fn


def compute_loss(batch, model, device, criterion_ce=None, loss_type="kl"):
    tokenized, labels, target_probs = batch
    tokenized = tokenized.to(device)
    labels = labels.to(device)
    target_probs = target_probs.to(device)

    logits = model(tokenized)
    if criterion_ce is not None:
        loss = criterion_ce(logits, labels)
    else:
        if loss_type == "kl":
            loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(logits, dim=-1), target_probs)
        elif loss_type == "ce":
            # cross-entropy with soft targets
            loss = torch.sum(target_probs * torch.log_softmax(logits, dim=-1), dim=-1).mean()

    return loss


def evaluate(model: MiniLM, device, criterion_ce, max_eval_samples=-1, **kwargs):
    objective_mode = kwargs["objective_mode"]
    past_time_steps = kwargs["past_time_steps"]
    future_time_steps = kwargs["future_time_steps"]
    n_shot = kwargs["n_shot"]
    latency_weight = kwargs["latency_weight"]
    example_filepath_list = kwargs["example_filepath_list"]
    eval_data_directory_path = kwargs["eval_data_directory_path"]
    loss_type = kwargs["loss_type"]

    model.eval()
    total_loss = 0

    # Load dataset
    datafile_paths = []
    file_names = [f for f in os.listdir(eval_data_directory_path) if f.endswith(".json")]
    for filename in file_names:
        datafile_paths.append(os.path.join(eval_data_directory_path, filename))

    eval_dataset = WADataset(
        filepath_list=datafile_paths,
        n_past_steps=past_time_steps,
        n_future_steps=future_time_steps,
        objective_mode=objective_mode,
        n_shot=n_shot,
        latency_weight=latency_weight,
        example_filepath_list=example_filepath_list,
    )

    # Select samples evenly as evaluation set
    n_samples = min(max_eval_samples, len(eval_dataset)) if max_eval_samples > 0 else len(eval_dataset)
    indices = np.arange(0, n_samples)
    eval_dataset = torch.utils.data.Subset(eval_dataset, indices)

    eval_dataloader = DataLoader(eval_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn(model))

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            loss = compute_loss(batch, model, device, criterion_ce, loss_type)
            total_loss += loss.item()

    # Calculate evaluation metrics
    avg_loss = total_loss / len(eval_dataloader)

    return {"loss": avg_loss}


def main(
    model_name,
    lr,
    weight_decay,
    n_epoch,
    batch_size,
    grad_accum_steps,
    rank,
    with_head,
    with_adapter,
    n_eval_samples,
    **kwargs,
):
    objective_mode = kwargs["objective_mode"]
    past_time_steps = kwargs["past_time_steps"]
    future_time_steps = kwargs["future_time_steps"]
    n_shot = kwargs["n_shot"]
    latency_weight = kwargs["latency_weight"]
    wa_head_type = kwargs["wa_head_type"]
    label_smoothing = kwargs.get("label_smoothing", 0.0)
    loss_type = kwargs.get("loss_type", "unknown")

    base_model_name = model_name.split("/")[-1]
    if kwargs["new_model_name"] is not None:
        new_model = kwargs["new_model_name"]
    else:
        new_model = f"{base_model_name}-wa-sft-{datetime.now().strftime('%m%d-%H%M%S')}"
    new_model_path = os.path.join("output", new_model)
    os.makedirs(new_model_path, exist_ok=True)

    # Load dataset
    data_directory_paths = [f"data/wa_processed_context_logs/collection_{i}" for i in range(30, 38)]
    datafile_paths = []
    for data_directory_path in data_directory_paths:
        file_names = [f for f in os.listdir(data_directory_path) if f.endswith(".json")]
        for filename in file_names:
            datafile_paths.append(os.path.join(data_directory_path, filename))

    example_data_directory = "data/wa_processed_context_logs/collection_39"
    example_datafile_paths = [
        os.path.join(example_data_directory, f) for f in os.listdir(example_data_directory) if f.endswith(".json")
    ]
    kwargs["example_filepath_list"] = example_datafile_paths
    eval_data_directory_path = "data/wa_processed_context_logs/collection_38"
    kwargs["eval_data_directory_path"] = eval_data_directory_path

    dataset = WADataset(
        filepath_list=datafile_paths,
        n_past_steps=past_time_steps,
        n_future_steps=future_time_steps,
        objective_mode=objective_mode,
        n_shot=n_shot,
        latency_weight=latency_weight,
        example_filepath_list=example_datafile_paths,
    )
    labels = [example["opt_parameters"] for example in dataset]
    label_counts = Counter(labels)
    print(f"Label distribution: {label_counts}")
    print(dataset[0]["prompt"])

    # Initialize model and tokenizer
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Enable memory efficient attention and gradient checkpointing
    model = MiniLM.from_pretrained(
        model_name,
        device=device,
        rank=rank,
        with_head=with_head,
        with_adapter=with_adapter,
        default_task="wa",
        wa_head_type=wa_head_type,
    ).to(device)

    # Enable gradient checkpointing
    # model.lm.gradient_checkpointing_enable()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(model),
        pin_memory=True,  # Enable pinned memory for faster data transfer
    )

    # Define the loss function
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if loss_type == "unknown" else None

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = math.ceil(n_epoch * len(dataloader) / grad_accum_steps)
    total_examples = len(dataset)

    # Cosine lr scheduler with warmup
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Use wandb for tracing the training process
    if with_head and with_adapter:
        project = "MiniLM-WA-SFT-Head-Adapter"
    elif with_head and not with_adapter:
        project = "MiniLM-WA-SFT-Head"
    elif not with_head and with_adapter:
        project = "MiniLM-WA-SFT-Adapter"
    else:
        project = "MiniLM-WA-SFT"
    wandb.init(
        project=project,
        name=new_model,
        config={
            "model": model_name,
            "train_data_dir": data_directory_paths,
            "eval_data_dir": eval_data_directory_path,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epoch": n_epoch,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "rank": rank,
            "objective_mode": objective_mode,
            "n_examples": total_examples,
            "past_time_steps": past_time_steps,
            "future_time_steps": future_time_steps,
            "with_head": with_head,
            "with_adapter": with_adapter,
            "n_shot": n_shot,
            "latency_weight": latency_weight,
            "wa_head_type": wa_head_type,
            "label_smoothing": label_smoothing,
            "loss_type": loss_type,
        },
    )
    print(f"Training {new_model} with head: {with_head}, with adapter: {with_adapter}")
    print(f"Total examples: {total_examples}, total steps: {total_steps}")

    # Add early stopping parameters
    patience = 3  # Number of epochs to wait for improvement
    min_eval_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(n_epoch), desc="Training"):
        model.train()

        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        accum_loss = 0

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            loss = compute_loss(batch, model, device, criterion_ce, loss_type)
            total_loss += loss.item()
            # Normalize loss
            residual = len(dataloader) % grad_accum_steps
            if residual != 0 and step >= len(dataloader) - residual:
                loss = loss / residual
            else:
                loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Perform gradient accumulation update
            if (step + 1) % grad_accum_steps == 0 or step == len(dataloader) - 1:
                wandb.log({"loss/step": accum_loss, "lr": optimizer.param_groups[0]["lr"]})
                accum_loss = 0
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if (step + 1) % (grad_accum_steps * 100) == 0 or step == len(dataloader) - 1:
                # Evaluate
                eval_metrics = evaluate(model, device, criterion_ce, max_eval_samples=n_eval_samples, **kwargs)
                current_eval_loss = eval_metrics["loss"]
                wandb.log({"loss/eval": current_eval_loss})
                model.train()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{n_epoch}, Average Loss: {avg_loss:.4f}, Eval Loss: {current_eval_loss:.4f}")
        wandb.log({"loss/train": avg_loss})

    # Finish wandb monitor
    wandb.finish()

    # Save model
    if patience_counter < patience:
        model.save(new_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for the script.")

    # Define the command-line arguments
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Set the model name"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Set the learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Set the weight decay")
    parser.add_argument("--epoch", type=int, default=5, help="Set the number of epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Set the batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Set the gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=128, help="Set the rank of the LoRA adapter")
    parser.add_argument("--past_time_steps", type=int, default=10, help="Set the number of past time steps")
    parser.add_argument("--future_time_steps", type=int, default=1, help="Set the number of future time steps")
    parser.add_argument("--n_shot", type=int, default=0, help="Set the number of shot")
    parser.add_argument("--latency_weight", type=float, default=10, help="Set the latency weight")
    parser.add_argument("--objective_mode", type=str, default="minimal", help="Set the objective mode")
    parser.add_argument("--with_head", action="store_true", help="Set whether to use the classification head")
    parser.add_argument(
        "--no_adapter",
        dest="with_adapter",
        action="store_false",
        help="Disable task-specific LoRA adapter (enabled by default)",
    )
    parser.add_argument("--n_eval_samples", type=int, default=1000, help="Set the number of evaluation samples")
    parser.add_argument("--new_model_name", type=str, default=None, help="Set the name of the new model")
    parser.add_argument("--wa_head_type", type=str, default="linear", help="Set the type of the Wi-Fi Aware head")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Set the label smoothing")
    parser.add_argument("--loss_type", type=str, default="kl", help="Set the loss type")
    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(
        model_name=args.model,
        lr=args.lr,
        n_epoch=args.epoch,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        rank=args.rank,
        with_head=args.with_head,
        with_adapter=args.with_adapter,
        n_eval_samples=args.n_eval_samples,
        past_time_steps=args.past_time_steps,
        future_time_steps=args.future_time_steps,
        n_shot=args.n_shot,
        latency_weight=args.latency_weight,
        objective_mode=args.objective_mode,
        new_model_name=args.new_model_name,
        wa_head_type=args.wa_head_type,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
    )
