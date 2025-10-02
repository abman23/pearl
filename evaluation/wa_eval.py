"""Inference script for PEARL and PEARL-Lite."""
import argparse
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.PEARL import PEARL
from model.PEARL_MPS import PEARL_MPS
from preprocess.utils import index_to_parameter, parameter_to_index
from preprocess.wa_dataset import WADataset

random.seed(42)


def main(
    eval_dir_name,
    example_dir_name,
    model_name,
    past_time_steps,
    future_time_steps,
    objective_mode,
    n_shot,
    latency_weight,
    sampling: bool,
    wandb_log: bool,
):
    n_examples_avg, n_inferences_avg, n_errors_avg, n_opt_avg, n_opt_mode_avg = [], [], [], [], []
    logits_mean, logits_std = [], []
    latency_scores_avg, overall_scores_avg, battery_scores_avg, reward_scores_avg = [], [], [], []
    latency_values_avg, battery_values_avg = [], []
    (
        upper_bound_latency_scores_avg,
        upper_bound_overall_scores_avg,
        upper_bound_battery_scores_avg,
        upper_bound_reward_scores_avg,
    ) = ([], [], [], [])
    upper_bound_latency_values_avg, upper_bound_battery_values_avg = [], []
    (
        lower_bound_latency_scores_avg,
        lower_bound_overall_scores_avg,
        lower_bound_battery_scores_avg,
        lower_bound_reward_scores_avg,
    ) = ([], [], [], [])
    lower_bound_latency_values_avg, lower_bound_battery_values_avg = [], []
    (
        heuristic_latency_scores_avg,
        heuristic_overall_scores_avg,
        heuristic_battery_scores_avg,
        heuristic_reward_scores_avg,
    ) = ([], [], [], [])
    heuristic_latency_values_avg, heuristic_battery_values_avg = [], []
    opt_reward_scores_avg, opt_latency_scores_avg, opt_overall_scores_avg, opt_battery_scores_avg = [], [], [], []
    opt_latency_values_avg, opt_battery_values_avg = [], []
    upper_bound_n_opt_avg, lower_bound_n_opt_avg, heuristic_n_opt_avg = [], [], []
    inf_time = []
    n_output_tokens_avg = []
    all_opt_param_count = defaultdict(int)
    all_llm_param_count = defaultdict(int)
    all_opt_mode_count = defaultdict(int)
    all_llm_mode_count = defaultdict(int)
    data_directory_path = f"data/wa_processed_context_logs/{eval_dir_name}"
    data_files = [f for f in os.listdir(data_directory_path) if f.endswith(".json")]
    default_parameter = parameter_to_index[("realtime", "bestEffort")]
    upper_bound_parameter = parameter_to_index[("realtime", "interactiveVoice")]
    lower_bound_parameter = parameter_to_index[("bulk", "background")]

    if wandb_log:
        wandb.init(
            project="PEARL-WA-Eval",
            name=model_name + "_" + eval_dir_name
            if n_shot == 0
            else model_name + "_" + eval_dir_name + "_" + str(n_shot),
            config={
                "model": model_name,
                "past_time_steps": past_time_steps,
                "future_time_steps": future_time_steps,
                "objective_mode": objective_mode,
                "n_shot": n_shot,
                "latency_weight": latency_weight,
            },
        )

    # Initialize PEARL model and tokenizer
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if device == torch.device("mps"):
        model = PEARL_MPS.from_pretrained(
            model_name, device=device, with_head=False, default_task="wa", load_wa_only=True
        ).to(device)
        print("Using HF model with MPS")
    else:
        model = PEARL.from_pretrained(
            model_name, device=device, with_head=False, default_task="wa", load_wa_only=True
        ).to(device)
        print("Using HF model with CUDA or CPU")
    tokenizer = model.tokenizer

    model.eval()

    example_data_directory = f"data/wa_processed_context_logs/{example_dir_name}"
    example_datafile_paths = [
        os.path.join(example_data_directory, f) for f in os.listdir(example_data_directory) if f.endswith(".json")
    ]

    if device == torch.device("cuda"):
        # Start memory tracking
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    for filename in data_files:
        # Construct full file path of each raw data file
        filepath = os.path.join(data_directory_path, filename)
        # Prepare the evaluation dataset
        dataset = WADataset(
            filepath_list=[filepath],
            n_past_steps=past_time_steps,
            n_future_steps=future_time_steps,
            objective_mode=objective_mode,
            n_shot=n_shot,
            latency_weight=latency_weight,
            example_filepath_list=example_datafile_paths,
        )
        # sample data points in sequence one by one
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"**Dataset**: {filename}, {len(dataset)} data points.\n")
        # print(dataset[0]["prompt"])

        # Evaluation
        n_inferences = 0
        n_errors = 0
        n_opt = 0
        dataset_logits = []
        latency_scores = 0.0
        overall_scores = 0.0
        battery_scores = 0.0
        reward_scores = 0.0
        latency_values = 0.0
        battery_values = 0.0
        opt_reward_scores = 0.0
        opt_latency_scores = 0.0
        opt_overall_scores = 0.0
        opt_battery_scores = 0.0
        opt_latency_values = 0.0
        opt_battery_values = 0.0
        upper_bound_latency_scores = 0.0
        upper_bound_overall_scores = 0.0
        upper_bound_battery_scores = 0.0
        upper_bound_reward_scores = 0.0
        upper_bound_latency_values = 0.0
        upper_bound_battery_values = 0.0
        upper_bound_n_opt = 0
        lower_bound_latency_scores = 0
        lower_bound_overall_scores = 0.0
        lower_bound_battery_scores = 0.0
        lower_bound_reward_scores = 0.0
        lower_bound_latency_values = 0.0
        lower_bound_battery_values = 0.0
        lower_bound_n_opt = 0
        heuristic_latency_scores = 0.0
        heuristic_overall_scores = 0.0
        heuristic_battery_scores = 0.0
        heuristic_reward_scores = 0.0
        heuristic_latency_values = 0.0
        heuristic_battery_values = 0.0
        heuristic_n_opt = 0
        n_output_tokens = []

        opt_param_count = defaultdict(int)
        llm_param_count = defaultdict(int)
        opt_mode_count = {"bulk": 0, "realtime": 0}
        llm_mode_count = {"bulk": 0, "realtime": 0}
        n_opt_mode = 0

        for example in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            opt_parameters = example["opt_parameters"].item()
            opt_standard_parameter = example["opt_standard_parameter"].item()
            opt_param_count[opt_parameters] += 1
            opt_mode = index_to_parameter[opt_standard_parameter][0]
            opt_mode_count[opt_mode] += 1

            prompt = example["prompt"]
            rewards = {param: reward.item() for param, reward in example["rewards"].items()}
            latencies = {param: latency.item() for param, latency in example["latencies"].items()}
            batteryUsages = {param: batteryUsage.item() for param, batteryUsage in example["batteryUsages"].items()}
            latency_values_dict = {
                param: latencyValue.item() for param, latencyValue in example["latency_values"].items()
            }
            battery_values_dict = {
                param: batteryValue.item() for param, batteryValue in example["battery_values"].items()
            }

            # PEARL model inference
            inf_start = time.time()
            inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(device)
            with torch.no_grad():
                prediction, logits = model.predict(
                    inputs["input_ids"], inputs["attention_mask"], task="wa", sampling=sampling
                )
            inf_end = time.time()
            inf_time.append(inf_end - inf_start)
            n_inferences += 1

            # Post-processing to extract the parameter
            if model.with_head:
                parameter = prediction.item()
                logits = logits.float().cpu().numpy()
                dataset_logits.append(logits)
                n_output_tokens.append(len(prediction))
            else:
                n_output_tokens.append(logits)
                # Regex for performanceMode
                perf_pattern = r"\b(bulk|realtime)\b"
                # Regex for accessCategory
                access_pattern = r"\b(bestEffort|background|interactiveVideo|interactiveVoice)\b"

                perf_matches = re.findall(perf_pattern, prediction[0])
                access_matches = re.findall(access_pattern, prediction[0])
                if perf_matches and access_matches:
                    performanceMode = perf_matches[0]
                    accessCategory = access_matches[0]
                    parameter_tuple = (performanceMode, accessCategory)
                    parameter = parameter_to_index.get(parameter_tuple, -1)

                    if parameter == -1:
                        n_errors += 1
                        parameter = default_parameter
                        print(
                            f"Error: {parameter_tuple} is not a valid combination, choose default parameter: {index_to_parameter[default_parameter]}"
                        )
                else:
                    n_errors += 1
                    parameter = default_parameter
                    print(
                        f"Error: No matched parameters found in the response, choose default parameter: {index_to_parameter[default_parameter]}"
                    )

            # Log statistics for evaluation
            llm_param_count[parameter] += 1
            llm_mode = index_to_parameter[parameter][0]
            llm_mode_count[llm_mode] += 1
            if llm_mode == opt_mode:
                n_opt_mode += 1
            latency = latencies[parameter]
            batteryUsage = batteryUsages[parameter]
            reward = rewards[parameter]
            overall = latency * latency_weight - batteryUsage

            if parameter == opt_standard_parameter:
                n_opt += 1

            latency_scores += latency
            battery_scores += batteryUsage
            reward_scores += reward
            overall_scores += overall
            latency_values += latency_values_dict[parameter]
            battery_values += battery_values_dict[parameter]

            opt_reward_scores += max(rewards.values())
            opt_latency_scores += latencies[opt_parameters]
            opt_overall_scores += latencies[opt_parameters] * latency_weight - batteryUsages[opt_parameters]
            opt_battery_scores += batteryUsages[opt_parameters]
            opt_latency_values += latency_values_dict[opt_parameters]
            opt_battery_values += battery_values_dict[opt_parameters]

            upper_bound_latency_scores += latencies[upper_bound_parameter]
            upper_bound_overall_scores += (
                latencies[upper_bound_parameter] * latency_weight - batteryUsages[upper_bound_parameter]
            )
            upper_bound_battery_scores += batteryUsages[upper_bound_parameter]
            upper_bound_reward_scores += rewards[upper_bound_parameter]
            upper_bound_latency_values += latency_values_dict[upper_bound_parameter]
            upper_bound_battery_values += battery_values_dict[upper_bound_parameter]
            if upper_bound_parameter == opt_standard_parameter:
                upper_bound_n_opt += 1

            lower_bound_latency_scores += latencies[lower_bound_parameter]
            lower_bound_overall_scores += (
                latencies[lower_bound_parameter] * latency_weight - batteryUsages[lower_bound_parameter]
            )
            lower_bound_battery_scores += batteryUsages[lower_bound_parameter]
            lower_bound_reward_scores += rewards[lower_bound_parameter]
            lower_bound_latency_values += latency_values_dict[lower_bound_parameter]
            lower_bound_battery_values += battery_values_dict[lower_bound_parameter]
            if lower_bound_parameter == opt_standard_parameter:
                lower_bound_n_opt += 1

            preferred_parameters = example["preferred_parameters"].item()
            heuristic_latency_scores += latencies[preferred_parameters]
            heuristic_overall_scores += (
                latencies[preferred_parameters] * latency_weight - batteryUsages[preferred_parameters]
            )
            heuristic_battery_scores += batteryUsages[preferred_parameters]
            heuristic_reward_scores += rewards[preferred_parameters]
            heuristic_latency_values += latency_values_dict[preferred_parameters]
            heuristic_battery_values += battery_values_dict[preferred_parameters]
            if preferred_parameters == opt_standard_parameter:
                heuristic_n_opt += 1

        # Log dataset-level statistics for evaluation
        n_cumulative_steps = n_inferences * future_time_steps
        n_examples_avg.append(n_cumulative_steps)
        n_inferences_avg.append(n_inferences)
        n_errors_avg.append(n_errors)
        n_opt_avg.append(n_opt)
        n_opt_mode_avg.append(n_opt_mode)
        latency_scores_avg.append(latency_scores / n_cumulative_steps)
        overall_scores_avg.append(overall_scores / n_cumulative_steps)
        battery_scores_avg.append(battery_scores / n_cumulative_steps)
        reward_scores_avg.append(reward_scores / n_cumulative_steps)
        latency_values_avg.append(latency_values / n_cumulative_steps)
        battery_values_avg.append(battery_values / n_cumulative_steps)

        upper_bound_n_opt_avg.append(upper_bound_n_opt)
        lower_bound_n_opt_avg.append(lower_bound_n_opt)
        heuristic_n_opt_avg.append(heuristic_n_opt)

        opt_reward_scores_avg.append(opt_reward_scores / n_cumulative_steps)
        opt_latency_scores_avg.append(opt_latency_scores / n_cumulative_steps)
        opt_overall_scores_avg.append(opt_overall_scores / n_cumulative_steps)
        opt_battery_scores_avg.append(opt_battery_scores / n_cumulative_steps)
        opt_latency_values_avg.append(opt_latency_values / n_cumulative_steps)
        opt_battery_values_avg.append(opt_battery_values / n_cumulative_steps)

        upper_bound_latency_scores_avg.append(upper_bound_latency_scores / n_cumulative_steps)
        upper_bound_overall_scores_avg.append(upper_bound_overall_scores / n_cumulative_steps)
        upper_bound_battery_scores_avg.append(upper_bound_battery_scores / n_cumulative_steps)
        upper_bound_reward_scores_avg.append(upper_bound_reward_scores / n_cumulative_steps)
        upper_bound_latency_values_avg.append(upper_bound_latency_values / n_cumulative_steps)
        upper_bound_battery_values_avg.append(upper_bound_battery_values / n_cumulative_steps)

        lower_bound_latency_scores_avg.append(lower_bound_latency_scores / n_cumulative_steps)
        lower_bound_overall_scores_avg.append(lower_bound_overall_scores / n_cumulative_steps)
        lower_bound_battery_scores_avg.append(lower_bound_battery_scores / n_cumulative_steps)
        lower_bound_reward_scores_avg.append(lower_bound_reward_scores / n_cumulative_steps)
        lower_bound_latency_values_avg.append(lower_bound_latency_values / n_cumulative_steps)
        lower_bound_battery_values_avg.append(lower_bound_battery_values / n_cumulative_steps)

        heuristic_latency_scores_avg.append(heuristic_latency_scores / n_cumulative_steps)
        heuristic_overall_scores_avg.append(heuristic_overall_scores / n_cumulative_steps)
        heuristic_battery_scores_avg.append(heuristic_battery_scores / n_cumulative_steps)
        heuristic_reward_scores_avg.append(heuristic_reward_scores / n_cumulative_steps)
        heuristic_latency_values_avg.append(heuristic_latency_values / n_cumulative_steps)
        heuristic_battery_values_avg.append(heuristic_battery_values / n_cumulative_steps)

        if model.with_head:
            logits_mean.append(np.mean(dataset_logits, axis=0))
            logits_std.append(np.std(dataset_logits, axis=0))

        n_output_tokens_avg.append(np.mean(n_output_tokens))

        print(
            f"average latency value: {latency_values_avg[-1]:.4f};\n"
            f"average battery value: {battery_values_avg[-1]:.4f};\n"
            f"average latency score: {latency_scores_avg[-1]:.4f};\n"
            f"average overall score: {overall_scores_avg[-1]:.4f};\n"
            f"average battery usage score: {battery_scores_avg[-1]:.4f};\n"
            f"average reward: {reward_scores_avg[-1]:.4f};\n"
            f"average opt reward: {opt_reward_scores_avg[-1]:.4f};\n"
        )
        print(f"LLM accuracy: {n_opt / n_inferences :.4f}")
        print(f"LLM mode accuracy: {n_opt_mode / n_inferences :.4f}")
        print(f"upper bound accuracy: {upper_bound_n_opt / n_inferences :.4f}")
        print(f"lower bound accuracy: {lower_bound_n_opt / n_inferences :.4f}")
        print(f"heuristic accuracy: {heuristic_n_opt / n_inferences :.4f}")
        print()

        for mode, count in opt_mode_count.items():
            print(f"number of optimal parameters in {mode} mode: {count}")
        for mode, count in llm_mode_count.items():
            print(f"number of parameters in {mode} mode chosen by LLM: {count}")
        print()

        for param in index_to_parameter.keys():
            all_opt_param_count[param] += opt_param_count[param]
            all_llm_param_count[param] += llm_param_count[param]

        for mode, count in opt_mode_count.items():
            all_opt_mode_count[mode] += count
        for mode, count in llm_mode_count.items():
            all_llm_mode_count[mode] += count

    print("\n********** Summary **********")
    print(f"Model name: {model_name}")
    print(
        f"n_shot: {n_shot}, latency weight: {latency_weight}, objective mode: {objective_mode}, past time steps: {past_time_steps}, future time steps: {future_time_steps}"
    )
    print(f"{round(time.time() - start, 2)} seconds used for evaluation.")
    print(f"Average inference time: {np.mean(inf_time) :.4f} s\n")
    print(
        f"Evaluation directory: {data_directory_path};\n"
        f"Totally {len(n_examples_avg)} evaluation time series;\n"
        f"average number of inferences made by LLM agent: {np.mean(n_inferences_avg)}, {n_inferences_avg};\n"
        f"average number of output tokens: {np.mean(n_output_tokens_avg)}, {n_output_tokens_avg};\n"
        f"average error rate: {np.sum(n_errors_avg) / np.sum(n_inferences_avg)};\n"
        f"average ratio of optimal parameters: {np.sum(n_opt_avg) / np.sum(n_inferences_avg)};\n"
        f"average ratio of optimal mode: {np.sum(n_opt_mode_avg) / np.sum(n_inferences_avg)};\n\n"
        f"average latency value: {np.mean(latency_values_avg)};\n"
        f"average battery value: {np.mean(battery_values_avg)};\n"
        f"average latency score: {np.mean(latency_scores_avg)};\n"
        f"average overall score: {np.mean(overall_scores_avg)};\n"
        f"average battery usage score: {np.mean(battery_scores_avg)};\n"
        f"average reward: {np.mean(reward_scores_avg)};\n\n"
        f"average opt latency value: {np.mean(opt_latency_values_avg)};\n"
        f"average opt battery value: {np.mean(opt_battery_values_avg)};\n"
        f"average opt latency score: {np.mean(opt_latency_scores_avg)};\n"
        f"average opt overall score: {np.mean(opt_overall_scores_avg)};\n"
        f"average opt battery usage score: {np.mean(opt_battery_scores_avg)};\n"
        f"average opt reward: {np.mean(opt_reward_scores_avg)};\n\n"
        f"average upper bound latency value: {np.mean(upper_bound_latency_values_avg)};\n"
        f"average upper bound battery value: {np.mean(upper_bound_battery_values_avg)};\n"
        f"average upper bound latency score: {np.mean(upper_bound_latency_scores_avg)};\n"
        f"average upper bound overall score: {np.mean(upper_bound_overall_scores_avg)};\n"
        f"average upper bound battery usage score: {np.mean(upper_bound_battery_scores_avg)};\n"
        f"average accuracy of upper bound: {np.sum(upper_bound_n_opt_avg) / np.sum(n_inferences_avg)};\n"
        f"average upper bound reward: {np.mean(upper_bound_reward_scores_avg)};\n\n"
        f"average lower bound latency value: {np.mean(lower_bound_latency_values_avg)};\n"
        f"average lower bound battery value: {np.mean(lower_bound_battery_values_avg)};\n"
        f"average lower bound latency score: {np.mean(lower_bound_latency_scores_avg)};\n"
        f"average lower bound overall score: {np.mean(lower_bound_overall_scores_avg)};\n"
        f"average lower bound battery usage score: {np.mean(lower_bound_battery_scores_avg)};\n"
        f"average accuracy of lower bound: {np.sum(lower_bound_n_opt_avg) / np.sum(n_inferences_avg)};\n"
        f"average lower bound reward: {np.mean(lower_bound_reward_scores_avg)};\n\n"
        f"average heuristic latency value: {np.mean(heuristic_latency_values_avg)};\n"
        f"average heuristic battery value: {np.mean(heuristic_battery_values_avg)};\n"
        f"average heuristic latency score: {np.mean(heuristic_latency_scores_avg)};\n"
        f"average heuristic overall score: {np.mean(heuristic_overall_scores_avg)};\n"
        f"average heuristic battery usage score: {np.mean(heuristic_battery_scores_avg)};\n"
        f"average accuracy of heuristic: {np.sum(heuristic_n_opt_avg) / np.sum(n_inferences_avg)};\n"
        f"average heuristic reward: {np.mean(heuristic_reward_scores_avg)};\n"
    )

    if model.with_head:
        print(f"\nAverage logits:\n{np.mean(logits_mean, axis=0)},\nstd:\n{np.mean(logits_std, axis=0)};")

    for param, count in all_opt_param_count.items():
        print(f"number of optimal parameters {index_to_parameter[param]}: {count}")
    for param, count in all_llm_param_count.items():
        print(f"number of parameters {index_to_parameter[param]} chosen by LLM: {count}")
    print()
    for mode, count in all_opt_mode_count.items():
        print(f"number of optimal parameters in {mode} mode: {count}")
    for mode, count in all_llm_mode_count.items():
        print(f"number of parameters in {mode} mode chosen by LLM: {count}")
    print()

    # Print peak memory usage
    if device == torch.device("cuda"):
        mem = torch.cuda.max_memory_reserved() / 1e9
        print(f"Peak Memory Usage: {mem:.02f} GB")

    if wandb_log:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for the script.")

    # Define the command-line arguments
    parser.add_argument("--eval_dir_name", type=str, default="collection_38", help="Set the evaluation directory name")
    parser.add_argument("--example_dir_name", type=str, default="collection_39", help="Set the example directory name")
    parser.add_argument("--past_time_steps", type=int, default=10, help="Set the past time steps")
    parser.add_argument(
        "--future_time_steps", type=int, default=1, help="Set the future time steps used for evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.2-3B",  # output/Llama-3.2-3B-thr-sft unsloth/Llama-3.2-3B
        help="Set the model name",
    )
    parser.add_argument("--objective_mode", type=str, default="joint", help="Set the objective mode")
    parser.add_argument("--n_shot", type=int, default=0, help="Set the number of shot")
    parser.add_argument("--latency_weight", type=float, default=10, help="Set the latency weight")
    parser.add_argument("--sampling", action="store_true", help="Use nucleus sampling to generate the response")
    parser.add_argument("--wandb", action="store_true", help="Use wandb to log the results or not")
    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(
        eval_dir_name=args.eval_dir_name,
        example_dir_name=args.example_dir_name,
        model_name=args.model,
        past_time_steps=args.past_time_steps,
        future_time_steps=args.future_time_steps,
        objective_mode=args.objective_mode,
        n_shot=args.n_shot,
        latency_weight=args.latency_weight,
        sampling=args.sampling,
        wandb_log=args.wandb,
    )
