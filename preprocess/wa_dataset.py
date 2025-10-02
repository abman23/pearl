import json
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

from preprocess.utils import (
    application_profiles,
    index_to_parameter,
    parameter_to_index,
)
from preprocess.wa_prompt import (
    prompt_battery,
    prompt_joint,
    prompt_latency,
    prompt_minimal,
)


def process_contexts(
    preprocessed_filepath_list: list[str], n_past_steps: int, n_future_steps: int
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Extract the contexts from a list of context log files and build an index mapping.

    :param preprocessed_filepath_list: list of preprocessed context log file paths with synthetic application types
    :param n_past_steps: number of past steps for prompt
    :param n_future_steps: number of future steps for optimal parameter calculation
    :return: (file_contexts, index_map) where index_map is a list of (file_idx, log_idx)
    """
    index_map = []
    file_contexts = []
    for file_idx, context_log_path in enumerate(preprocessed_filepath_list):
        contexts_dict = json.load(open(context_log_path))
        n_valid_logs = len(
            [applicationTypes for applicationTypes in contexts_dict["applicationTypes"] if applicationTypes != []]
        )
        for key in contexts_dict.keys():
            contexts_dict[key] = contexts_dict[key][:n_valid_logs]
        file_contexts.append(contexts_dict)
        n_logs = len(file_contexts[file_idx]["timestamp"])
        # print(f"n_logs: {n_logs} for file: {context_log_path}")
        # for log_idx in range(n_past_steps, min(n_logs - n_future_steps + 1, 200)):
        for log_idx in range(n_past_steps, n_logs - n_future_steps + 1):
            index_map.append((file_idx, log_idx))

    return file_contexts, index_map


class WADataset(Dataset):
    def __init__(
        self,
        filepath_list: list[str],
        n_past_steps: int = 10,
        n_future_steps: int = 30,
        objective_mode: str = "joint",
        n_shot: int = 0,
        latency_weight: float = 1.5,
        example_filepath_list: list[str] = ["./data/wa_processed_context_logs/example_context.json"],
    ):
        file_contexts, index_map = process_contexts(sorted(filepath_list), n_past_steps, n_future_steps)
        self.file_contexts = file_contexts
        self.index_map = index_map
        # self.parameters_metrics = json.load(open("./data/wa_metrics_new.json"))
        self.parameters_metrics = json.load(open("./data/wa_processed_context_logs/parameters_metrics_new.json"))
        self.objective_mode = objective_mode
        # for param, metrics in self.parameters_metrics.items():
        #     print(f"param: {index_to_parameter[int(param)]}, metrics: {metrics}")
        self.n_past_steps = n_past_steps
        self.n_future_steps = n_future_steps
        self.n_shot = n_shot
        self.latency_weight = latency_weight
        self.example_file_contexts_list = []
        self.example_index_map_list = []

        if n_shot > 0:
            self.n_example_datasets = len(example_filepath_list)
            self.example_dataset_index = 0
            for example_filepath in sorted(example_filepath_list):
                example_file_contexts, example_index_map = process_contexts(
                    [example_filepath], n_past_steps, n_future_steps
                )
                self.example_file_contexts_list.append(example_file_contexts)
                self.example_index_map_list.append(example_index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, log_idx = self.index_map[idx]
        file_ctx = self.file_contexts[file_idx]
        context_logs, sender_device, receiver_device = self.format_context_logs_with_examples(file_ctx, log_idx)

        if self.objective_mode == "joint":
            prompt = prompt_joint.format(
                context_logs=context_logs, sender_device=sender_device, receiver_device=receiver_device
            )
        elif self.objective_mode == "minimal":
            prompt = prompt_minimal.format(context_logs=context_logs)
        elif self.objective_mode == "latency":
            prompt = prompt_latency.format(
                context_logs=context_logs, sender_device=sender_device, receiver_device=receiver_device
            )
        elif self.objective_mode == "battery":
            prompt = prompt_battery.format(
                context_logs=context_logs, sender_device=sender_device, receiver_device=receiver_device
            )
        elif self.objective_mode == "one_side":
            prompt = prompt_joint.format(
                context_logs=context_logs, sender_device=sender_device, receiver_device=receiver_device
            )
        else:  # baseline
            prompt = prompt_joint.format(
                context_logs=context_logs, sender_device=sender_device, receiver_device=receiver_device
            )

        (
            opt_parameters,
            preferred_parameters,
            rewards,
            latencies,
            batteryUsages,
            opt_standard_parameter,
            latency_values,
            battery_values,
        ) = self.calculate_opt_parameters(file_ctx, log_idx)

        return {
            "prompt": prompt,
            "opt_parameters": opt_parameters,
            "preferred_parameters": preferred_parameters,
            "rewards": rewards,
            "latencies": latencies,
            "batteryUsages": batteryUsages,
            "opt_standard_parameter": opt_standard_parameter,
            "latency_values": latency_values,
            "battery_values": battery_values,
        }

    def format_context_logs_with_examples(self, ctx: dict[list], log_idx: int):
        if self.objective_mode == "joint" or self.objective_mode == "minimal":
            ctx_head = "| Time of day | Application | Receiver Battery Level | Sender Battery Level |"
        elif self.objective_mode == "latency":
            ctx_head = "| Time of day | Application |"
        elif self.objective_mode == "battery":
            ctx_head = "| Receiver Battery Level | Sender Battery Level |"
        elif self.objective_mode == "one_side":
            ctx_head = "| Time of day | Application | Receiver Battery Level |"
        else:  # baseline
            ctx_head = "| Time of day | Application | Receiver Battery Level | Sender Battery Level |"
        context_logs, sender_device, receiver_device = self.format_context_logs(ctx, log_idx, ctx_head)

        if self.n_shot > 0:
            additional_context_texts = []
            for i in range(self.n_shot):
                additional_context_texts.append(f"\nExample {i+1}:")
                additional_context_texts.append(ctx_head)

                # iterate through example datasets in sequence
                dataset_index = self.example_dataset_index
                self.example_dataset_index = (self.example_dataset_index + 1) % self.n_example_datasets
                example_index_map = self.example_index_map_list[dataset_index]
                example_index = log_idx % len(example_index_map)  # select example index in sequence
                example_file_idx, example_log_idx = example_index_map[example_index]
                example_file_ctx = self.example_file_contexts_list[dataset_index][example_file_idx]

                example_context_logs, _, _ = self.format_context_logs(example_file_ctx, example_log_idx)
                additional_context_texts.append(example_context_logs)

                example_opt_parameters, _, _, _, _, _, _, _ = self.calculate_opt_parameters(
                    example_file_ctx, example_log_idx
                )
                example_opt_parameters = index_to_parameter[example_opt_parameters]
                additional_context_texts.append(f"Optimal (performanceMode, accessCategory)= {example_opt_parameters}")

            context_logs = "\n".join(additional_context_texts) + "\n" + context_logs

        return context_logs, sender_device, receiver_device

    def format_context_logs(self, ctx: dict[list], log_idx: int, ctx_head: str = "") -> str:
        ctx_logs = [ctx_head] if ctx_head else []
        for i in range(log_idx - self.n_past_steps, log_idx):
            if self.objective_mode == "joint" or self.objective_mode == "minimal":
                ctx_log = f"| {ctx['timestamp'][i]} | {ctx['applicationTypes'][i]} | {ctx['localBattery'][i]} | {ctx['remoteBattery'][i]} |"
            elif self.objective_mode == "latency":
                ctx_log = f"| {ctx['timestamp'][i]} | {ctx['applicationTypes'][i]} |"
            elif self.objective_mode == "battery":
                ctx_log = f"| {ctx['localBattery'][i]} | {ctx['remoteBattery'][i]} |"
            elif self.objective_mode == "one_side":
                ctx_log = f"| {ctx['timestamp'][i]} | {ctx['applicationTypes'][i]} | {ctx['localBattery'][i]} |"
            else:  # baseline
                ctx_log = f"| {ctx['timestamp'][i]} | {ctx['applicationTypes'][i]} | {ctx['localBattery'][i]} | {ctx['remoteBattery'][i]} |"
            ctx_logs.append(ctx_log)

        sender_device = ctx["remoteDevice"][log_idx]
        receiver_device = ctx["localDevice"][log_idx]
        return "\n".join(ctx_logs), sender_device, receiver_device

    def calculate_opt_parameters(self, ctx: dict[list], log_idx: int):
        futureApplicationTypes = ctx["applicationTypes"][log_idx : log_idx + self.n_future_steps]
        futureSenderBatteries = ctx["remoteBattery"][log_idx : log_idx + self.n_future_steps]
        futureReceiverBatteries = ctx["localBattery"][log_idx : log_idx + self.n_future_steps]
        if "latency" in ctx.keys():
            futureLatencies = ctx["latency"][log_idx : log_idx + self.n_future_steps]
        else:
            futureLatencies = [0] * self.n_future_steps
        if "batteryUsage" in ctx.keys():
            futureBatteryUsages = ctx["batteryUsage"][log_idx : log_idx + self.n_future_steps]
        else:
            futureBatteryUsages = [0] * self.n_future_steps

        rewards = defaultdict(float)
        latencies = defaultdict(float)
        batteryUsages = defaultdict(float)
        standard_rewards = defaultdict(float)
        latency_values = defaultdict(float)
        battery_values = defaultdict(float)

        for index in index_to_parameter.keys():
            metrics_dict = self.parameters_metrics[str(index)]

            # add up the rewards for all future steps
            for applicationTypes, senderBattery, receiverBattery, latency, batteryUsage in zip(
                futureApplicationTypes,
                futureSenderBatteries,
                futureReceiverBatteries,
                futureLatencies,
                futureBatteryUsages,
            ):
                # battery penalty
                # battery_usage = metrics_dict["batteryUsage"]
                battery_usage = metrics_dict["batteryUsage"] if batteryUsage == 0 else batteryUsage[str(index)]
                battery_score = battery_usage * (1 / senderBattery + 1 / receiverBattery) / 2

                # latency reward
                latency_scores = []
                latency_value = metrics_dict["latency"] if latency == 0 else latency[str(index)]
                # latency_value = metrics_dict["latency"]
                for applicationType in applicationTypes:
                    profile = application_profiles[applicationType]
                    score = max(0, 1 - (latency_value / profile["latencyTolerance"]))
                    latency_scores.append(score)

                latency_score = np.mean(latency_scores)

                # calculate reward
                standard_reward = self.latency_weight * latency_score - battery_score
                if self.objective_mode in ["joint", "minimal"]:
                    reward = standard_reward
                elif self.objective_mode == "latency":
                    reward = latency_score
                elif self.objective_mode == "battery":
                    reward = -battery_score
                elif self.objective_mode == "one_side":
                    battery_score_one_side = battery_usage * (1 / receiverBattery)
                    reward = self.latency_weight * latency_score - battery_score_one_side
                else:  # baseline
                    reward = -latency_value - battery_usage

                rewards[index] += reward
                latencies[index] += latency_score
                batteryUsages[index] += battery_score
                standard_rewards[index] += standard_reward
                latency_values[index] += latency_value
                battery_values[index] += battery_usage
        # rewards, latencies, batteryUsages = calculate_rewards(futureApplicationTypes, futureSenderBatteries, futureReceiverBatteries, latency_weight=latency_weight, battery_weight=battery_weight, parameters_metrics=self.parameters_metrics, objective_mode=self.objective_mode)
        opt_parameters = max(
            rewards, key=rewards.get
        )  # optimal parameterindex for current log_idx based on future application types
        opt_standard_parameter = max(standard_rewards, key=standard_rewards.get)

        # choose the mode of the preferred parameters based on past application types
        pastApplicationTypes = ctx["applicationTypes"][log_idx - self.n_past_steps : log_idx]
        preferred_parameters_count = defaultdict(int)
        for applicationTypes in pastApplicationTypes:
            for applicationType in applicationTypes:
                profile = application_profiles[applicationType]
                preferred_parameters_count[profile["preferredParameter"]] += 1
        preferred_parameters = parameter_to_index[max(preferred_parameters_count, key=preferred_parameters_count.get)]

        return (
            opt_parameters,
            preferred_parameters,
            rewards,
            latencies,
            batteryUsages,
            opt_standard_parameter,
            latency_values,
            battery_values,
        )


if __name__ == "__main__":
    # export PYTHONPATH="$PYTHONPATH:$(pwd)"
    dataset = WADataset(
        filepath_list=["data/wa_processed_context_logs/train/contexts_morning_low_battery.json"],
        objective_mode="baseline",
    )
    print(dataset[0]["prompt"])
    rewards = dataset[0]["rewards"]
    print("\nRewards:")
    for index, parameter in index_to_parameter.items():
        print(f"{parameter}: {rewards[index]}")
