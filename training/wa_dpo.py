"""DPO for PEARL model on Wi-Fi Aware parameter tuning.

"""
import argparse
import math
import os.path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from model.PEARL import PEARL
from preprocess.utils import index_to_parameter
from preprocess.wa_dataset import WADataset


def collate_fn(model):
    def _collate_fn(batch):
        # Preprocess the prompts and labels
        prompts = [example["prompt"] for example in batch]
        rewards_list = [example["rewards"] for example in batch]
        rejected_parameters = [min(rewards, key=rewards.get) for rewards in rewards_list]

        if model.with_head:
            tokenized = model.tokenizer(text=prompts, padding=True, add_special_tokens=True, return_tensors="pt")
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            chosen_params = torch.tensor([example["opt_parameters"] for example in batch])
            rejected_params = torch.tensor(rejected_parameters)

            return (
                input_ids,
                attention_mask,
                chosen_params,
                rejected_params,
            )
        else:
            # Construct input_ids, attention_mask for chosen and rejected completions
            chosen_input_list, rejected_input_list = [], []
            prompt_list = []
            for example, rejected_param_id in zip(batch, rejected_parameters):
                prompt = example["prompt"]
                prompt_list.append(prompt)
                opt_parameters = index_to_parameter[example["opt_parameters"]]
                chosen_label = f" ({opt_parameters[0]}, {opt_parameters[1]}){model.tokenizer.eos_token}"
                rejected_parameters = index_to_parameter[rejected_param_id]
                rejected_label = f" ({rejected_parameters[0]}, {rejected_parameters[1]}){model.tokenizer.eos_token}"
                chosen_input_texts = prompt + chosen_label
                rejected_input_texts = prompt + rejected_label
                chosen_input_list.append(chosen_input_texts)
                rejected_input_list.append(rejected_input_texts)

            prompt_encodings = model.tokenizer(
                prompt_list, padding=True, padding_side="left", add_special_tokens=False, return_tensors="pt"
            )
            prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1)

            chosen_encodings = model.tokenizer(
                chosen_input_list, padding=True, padding_side="left", add_special_tokens=False, return_tensors="pt"
            )
            rejected_encodings = model.tokenizer(
                rejected_input_list, padding=True, padding_side="left", add_special_tokens=False, return_tensors="pt"
            )

            return (
                chosen_encodings.input_ids,
                chosen_encodings.attention_mask,
                rejected_encodings.input_ids,
                rejected_encodings.attention_mask,
                prompt_lengths,
            )

    return _collate_fn


# DPO loss calculation is adapted from https://github.com/0xallam/Direct-Preference-Optimization/blob/main/src/train.py
def _get_log_probs(logits, labels, prompt_lengths):
    """
    Calculate the sum of log probabilities of the response tokens.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        prompt_lengths: (batch_size)

    Returns:
        response_log_probs: (batch_size)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    # print(labels.shape)
    response_mask = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()
    response_log_probs_sum = (token_log_probs * response_mask).sum(dim=-1)

    return response_log_probs_sum


def preference_loss(chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs, beta=0.1):
    chosen_relative_log_probs = chosen_log_probs - ref_chosen_log_probs
    rejected_relative_log_probs = rejected_log_probs - ref_rejected_log_probs

    loss = -F.logsigmoid(beta * (chosen_relative_log_probs - rejected_relative_log_probs))
    chosen_rewards = chosen_relative_log_probs.detach()
    rejected_rewards = rejected_relative_log_probs.detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    reward_margins = chosen_rewards - rejected_rewards

    # chosen/rejected rewards are not scaled by beta
    return loss, chosen_rewards, rejected_rewards, reward_accuracies, reward_margins


def compute_loss(batch, model, ref_model, device, beta, loss_type="ce"):
    if model.with_head:
        input_ids, attention_mask, chosen_params, rejected_params = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        chosen_params = chosen_params.to(device)
        rejected_params = rejected_params.to(device)

        logits = model(input_ids, attention_mask).unsqueeze(1)
        ref_logits = ref_model(input_ids, attention_mask).unsqueeze(1)
        chosen_labels = chosen_params.unsqueeze(1)
        rejected_labels = rejected_params.unsqueeze(1)
        prompt_lengths = torch.zeros(chosen_labels.shape[0], device=chosen_labels.device)

        chosen_log_probs = _get_log_probs(logits, chosen_labels, prompt_lengths=prompt_lengths)
        rejected_log_probs = _get_log_probs(logits, rejected_labels, prompt_lengths=prompt_lengths)
        ref_chosen_log_probs = _get_log_probs(ref_logits, chosen_labels, prompt_lengths=prompt_lengths)
        ref_rejected_log_probs = _get_log_probs(ref_logits, rejected_labels, prompt_lengths=prompt_lengths)
    else:
        chosen_ids, chosen_attention_mask, rejected_ids, rejected_attention_mask, prompt_lengths = batch
        chosen_ids = chosen_ids.to(device)
        chosen_attention_mask = chosen_attention_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_attention_mask = rejected_attention_mask.to(device)
        prompt_lengths = prompt_lengths.to(device)

        chosen_log_probs = _get_log_probs(model(chosen_ids, chosen_attention_mask).logits, chosen_ids, prompt_lengths)
        rejected_log_probs = _get_log_probs(
            model(rejected_ids, rejected_attention_mask).logits, rejected_ids, prompt_lengths
        )
        ref_chosen_log_probs = _get_log_probs(
            ref_model(chosen_ids, chosen_attention_mask).logits, chosen_ids, prompt_lengths
        )
        ref_rejected_log_probs = _get_log_probs(
            ref_model(rejected_ids, rejected_attention_mask).logits, rejected_ids, prompt_lengths
        )

    loss, chosen_reward, rejected_reward, reward_accuracies, reward_margins = preference_loss(
        chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs, beta
    )
    return loss, chosen_reward, rejected_reward, reward_accuracies, reward_margins


def evaluate(model: PEARL, ref_model: PEARL, device, max_eval_samples=-1, **kwargs):
    objective_mode = kwargs["objective_mode"]
    past_time_steps = kwargs["past_time_steps"]
    future_time_steps = kwargs["future_time_steps"]
    n_shot = kwargs["n_shot"]
    latency_weight = kwargs["latency_weight"]
    example_filepath_list = kwargs["example_filepath_list"]
    eval_data_directory_path = kwargs["eval_data_directory_path"]
    beta = kwargs["beta"]
    loss_type = kwargs["loss_type"]

    model.eval()
    model.lm.config.use_cache = True
    total_loss = []
    chosen_rewards = []
    rejected_rewards = []
    reward_accuracies = []
    reward_margins = []

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
            loss, chosen_reward, rejected_reward, reward_accuracy, reward_margin = compute_loss(
                batch, model, ref_model, device, beta, loss_type
            )
            total_loss.append(loss.cpu().numpy().tolist())
            chosen_rewards.append(chosen_reward.cpu().numpy().tolist())
            rejected_rewards.append(rejected_reward.cpu().numpy().tolist())
            reward_accuracies.append(reward_accuracy.cpu().numpy().tolist())
            reward_margins.append(reward_margin.cpu().numpy().tolist())

    # Calculate evaluation metrics
    avg_loss = np.mean(total_loss)
    avg_chosen_reward = np.mean(chosen_rewards)
    avg_rejected_reward = np.mean(rejected_rewards)
    avg_reward_accuracy = np.mean(reward_accuracies)
    avg_reward_margin = np.mean(reward_margins)

    return {
        "loss": avg_loss,
        "chosen_reward": avg_chosen_reward,
        "rejected_reward": avg_rejected_reward,
        "reward_accuracy": avg_reward_accuracy,
        "reward_margin": avg_reward_margin,
    }


def main(
    model_name,
    beta,
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
    kwargs["beta"] = beta
    wa_head_type = kwargs["wa_head_type"]
    loss_type = kwargs["loss_type"]

    if kwargs["new_model_name"] is not None:
        new_model = kwargs["new_model_name"]
    else:
        base_model_name = model_name.split("/")[-1]
        new_model = f"{base_model_name}-wa-dpo-{datetime.now().strftime('%m%d-%H%M%S')}"
    new_model_path = os.path.join("output", new_model)
    os.makedirs(new_model_path, exist_ok=True)

    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable memory efficient attention and gradient checkpointing
    model = PEARL.from_pretrained(
        model_name,
        rank=rank,
        with_head=with_head,
        with_adapter=with_adapter,
        default_task="wa",
        wa_head_type=wa_head_type,
    ).to(device)
    ref_model = PEARL.from_pretrained(
        model_name,
        rank=rank,
        with_head=with_head,
        with_adapter=with_adapter,
        default_task="wa",
        wa_head_type=wa_head_type,
    ).to(device)

    # Disable gradients for reference model
    ref_model.eval()
    ref_model.requires_grad_(False)

    # Enable gradient checkpointing
    # model.lm.gradient_checkpointing_enable()

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
    print(dataset[0]["prompt"])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(model),
        pin_memory=True,  # Enable pinned memory for faster data transfer
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = math.ceil(n_epoch * len(dataloader) / grad_accum_steps)
    total_examples = len(dataset)

    # Cosine lr scheduler with warmup
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Use wandb for tracing the training process
    if model.with_head and model.with_adapter:
        project = "PEARL-WA-DPO-Head-Adapter"
    elif model.with_head and not model.with_adapter:
        project = "PEARL-WA-DPO-Head"
    elif not model.with_head and model.with_adapter:
        project = "PEARL-WA-DPO-Adapter"
    else:
        project = "PEARL-WA-DPO"
    wandb.init(
        project=project,
        name=new_model,
        config={
            "model": model_name,
            "train_data_dir": data_directory_paths,
            "eval_data_dir": eval_data_directory_path,
            "beta": beta,
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
            "with_head": model.with_head,
            "with_adapter": model.with_adapter,
            "n_shot": n_shot,
            "latency_weight": latency_weight,
            "wa_head_type": wa_head_type,
            "loss_type": loss_type,
        },
    )
    print(f"Training {new_model} with head: {model.with_head}, with adapter: {model.with_adapter}")
    print(f"Total examples: {total_examples}, total steps: {total_steps}")

    # Add early stopping parameters
    patience = 3  # Number of epochs to wait for improvement
    min_eval_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(n_epoch), desc="Training"):
        model.train()
        model.lm.config.use_cache = False

        total_loss = []
        chosen_rewards = []
        rejected_rewards = []
        reward_accuracies = []
        reward_margins = []
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        accum_loss = 0

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            loss, chosen_reward, rejected_reward, reward_accuracy, reward_margin = compute_loss(
                batch, model, ref_model, device, beta, loss_type
            )
            total_loss.append(loss.detach().cpu().numpy().tolist())
            chosen_rewards.append(chosen_reward.cpu().numpy().tolist())
            rejected_rewards.append(rejected_reward.cpu().numpy().tolist())
            reward_accuracies.append(reward_accuracy.cpu().numpy().tolist())
            reward_margins.append(reward_margin.cpu().numpy().tolist())
            loss = loss.mean()
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
                eval_metrics = evaluate(model, ref_model, device, max_eval_samples=n_eval_samples, **kwargs)
                wandb.log(
                    {
                        "loss/eval": eval_metrics["loss"],
                        "chosen_reward/eval": eval_metrics["chosen_reward"],
                        "rejected_reward/eval": eval_metrics["rejected_reward"],
                        "reward_accuracy/eval": eval_metrics["reward_accuracy"],
                        "reward_margin/eval": eval_metrics["reward_margin"],
                    }
                )

                model.train()

        avg_loss = np.mean(total_loss)
        avg_chosen_reward = np.mean(chosen_rewards)
        avg_rejected_reward = np.mean(rejected_rewards)
        avg_reward_accuracy = np.mean(reward_accuracies)
        avg_reward_margin = np.mean(reward_margins)
        print(
            f"Epoch {epoch+1}/{n_epoch}, Average Loss: {avg_loss:.4f}, Average Chosen Reward: {avg_chosen_reward:.4f}, Average Rejected Reward: {avg_rejected_reward:.4f}, Average Reward Accuracy: {avg_reward_accuracy:.4f}, Average Reward Margin: {avg_reward_margin:.4f}"
        )
        wandb.log(
            {
                "loss/train": avg_loss,
                "chosen_reward/train": avg_chosen_reward,
                "rejected_reward/train": avg_rejected_reward,
                "reward_accuracy/train": avg_reward_accuracy,
                "reward_margin/train": avg_reward_margin,
            }
        )

        if epoch < n_epoch - 1 and epoch % 5 == 0:
            model.save(new_model_path + f"/epoch_{epoch+1}")

    # Finish wandb monitor
    wandb.finish()

    # Save model
    if patience_counter < patience:
        model.save(new_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for the script.")

    # Define the command-line arguments
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B", help="Set the model name")
    parser.add_argument("--beta", type=float, default=0.1, help="Set the beta")
    parser.add_argument("--lr", type=float, default=1e-5, help="Set the learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Set the weight decay")
    parser.add_argument("--epoch", type=int, default=3, help="Set the number of epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Set the batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Set the gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=128, help="Set the rank of the LoRA adapter")
    parser.add_argument("--past_time_steps", type=int, default=10, help="Set the number of past time steps")
    parser.add_argument("--future_time_steps", type=int, default=1, help="Set the number of future time steps")
    parser.add_argument("--n_shot", type=int, default=0, help="Set the number of shot")
    parser.add_argument("--latency_weight", type=float, default=10.0, help="Set the latency weight")
    parser.add_argument("--objective_mode", type=str, default="joint", help="Set the objective mode")
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
    parser.add_argument("--loss_type", type=str, default="ce", help="Set the loss type")
    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(
        model_name=args.model,
        beta=args.beta,
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
        loss_type=args.loss_type,
    )
