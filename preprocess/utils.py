import copy
import random
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt

parameter_to_index = {
    ("bulk", "bestEffort"): 0,
    ("bulk", "background"): 1,
    ("realtime", "bestEffort"): 2,
    ("realtime", "interactiveVideo"): 3,
    ("realtime", "interactiveVoice"): 4,
    ("bulk", "interactiveVideo"): 5,
    ("bulk", "interactiveVoice"): 6,
    ("realtime", "background"): 7,
}
index_to_parameter = {v: k for k, v in parameter_to_index.items()}


application_profiles = {
    "textMessage": {
        "latencyTolerance": 200,
        "throughputRequirement": 0.01,
        "priorityWeight": 1.0,
        "preferredParameter": ("realtime", "bestEffort"),
    },
    "voiceChat": {
        "latencyTolerance": 50,
        "throughputRequirement": 0.064,
        "priorityWeight": 1.5,
        "preferredParameter": ("realtime", "interactiveVoice"),
    },
    "videoCall": {
        "latencyTolerance": 100,
        "throughputRequirement": 1.5,
        "priorityWeight": 2.0,
        "preferredParameter": ("realtime", "interactiveVideo"),
    },
    "sensorSync": {
        "latencyTolerance": 1000,
        "throughputRequirement": 0.05,
        "priorityWeight": 0.5,
        "preferredParameter": ("bulk", "background"),
    },
    "photoTransfer": {
        "latencyTolerance": 2000,
        "throughputRequirement": 10,
        "priorityWeight": 1.0,
        "preferredParameter": ("bulk", "bestEffort"),
    },
    "videoUpload": {
        "latencyTolerance": 5000,
        "throughputRequirement": 30,
        "priorityWeight": 1.5,
        "preferredParameter": ("bulk", "bestEffort"),
    },
    "firmwareUpdate": {
        "latencyTolerance": 10000,
        "throughputRequirement": 20,
        "priorityWeight": 1.0,
        "preferredParameter": ("bulk", "background"),
    },
    "mapSync": {
        "latencyTolerance": 500,
        "throughputRequirement": 0.5,
        "priorityWeight": 1.2,
        "preferredParameter": ("realtime", "bestEffort"),
    },
}


# Haversine formula to calculate distance between two points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# Constants
MOBILITY_THRESHOLD = 1.5  # m/s
WEAK_RSSI_THRESHOLD = -90  # RSSI lower bound
PEAK_HOUR_RANGES = [(9, 12), (13, 18)]  # Peak hour ranges
BATTERY_LOW_THRESHOLD = 30  # Battery capacity threshold
DENSITY_THRESHOLD = 10  # Network density threshold
TOP_NETWORKS_COUNT = 9  # Number of top networks to consider
ENVIRONMENT_THRESHOLDS = {"outdoor": -70, "indoor": -60}  # RSSI thresholds for environment classification


# Calculate speeds and classify mobility
def classify_mobility(locations, threshold=MOBILITY_THRESHOLD):
    d, t = 0, 0
    for i in range(1, len(locations)):  # lon, lat, time (seconds)
        lon1, lat1, t1 = locations[i - 1]
        lon2, lat2, t2 = locations[i]
        distance = haversine(lon1, lat1, lon2, lat2)
        time_diff = t2 - t1
        d += distance
        t += time_diff

    # Average speed
    avg_speed = d / t if t != 0 else 0
    # print(f"avg_speed: {avg_speed}")
    mobility = "high" if avg_speed > threshold else "low"
    return mobility, avg_speed


def classify_hours(timestamp: str, scenario: str = "office"):
    # Parse the timestamp string into a datetime object
    dt_object = datetime.strptime(timestamp, "%H:%M:%S")

    # Extract the hour
    hour = dt_object.hour
    t = dt_object.strftime("%H:%M")

    # Determine if it's peak or off-peak
    is_peak = any(start <= hour < end for start, end in PEAK_HOUR_RANGES)
    return "peak" if is_peak else "off-peak", t


def classify_battery(capacity: int):
    """Classify battery level as high or low.

    Args:
        capacity: Battery capacity percentage

    Returns:
        str: 'high' if capacity >= threshold, 'low' otherwise
    """
    return "high" if capacity >= BATTERY_LOW_THRESHOLD else "low"


def classify_env_and_density(networks: list, num_threshold: int = 10):
    """Determine environment scenario and AP density

    :param networks: scanned networks at one time step
    :param num_threshold: number of networks to be considered as dense
    :return: (env, density, sorted_pairs)
    """
    pairs = {}
    for network in networks:
        rssi = network["rssi"]
        bssid = network["bssid"]
        if rssi < WEAK_RSSI_THRESHOLD:
            continue

        pairs[bssid] = rssi

    pairs_sorted = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    k = min(TOP_NETWORKS_COUNT, len(pairs_sorted) - 1)
    rssi_top10 = pairs_sorted[k][1]
    outdoor_diff = abs(rssi_top10 + ENVIRONMENT_THRESHOLDS["outdoor"])
    indoor_diff = abs(rssi_top10 + ENVIRONMENT_THRESHOLDS["indoor"])
    env = "outdoor" if outdoor_diff < indoor_diff else "indoor"

    avail = "high" if len(pairs) >= num_threshold else "low"
    # print(f"n_bssids: {len(pairs)}")

    return env, avail, dict(pairs_sorted)


def find_stable_bssid(
    networks_window: list[list[dict]], rssi_threshold: int, thr_list=None
) -> tuple[list[str], dict, dict]:
    """
    Find BSSIDs that keep their RSSI values above a threshold for the most consecutive time steps starting from 0.
    :param thr_list: a list of RSSI thresholds in case you need different thresholds for the BSSID-RSSI pair in each time step.
    :param networks_window: networks lists ordered by timestep
    :param rssi_threshold: RSSI threshold
    :return: A list of the most stable BSSIDs, the counts of consecutive time steps of each initial BSSID, and their
    accumulated RSSI in the time window
    """
    # Dictionary to keep track of the maximum consecutive count for each BSSID
    consecutive_counts: dict[str:int] = {}
    accumulated_rssi: dict[str, int] = {}

    for timestep, networks in enumerate(networks_window):
        for network in networks:
            bssid, rssi = network["bssid"], network["rssi"]
            if rssi <= WEAK_RSSI_THRESHOLD:
                continue
            if timestep == 0:  # Initialize all BSSIDs at timestep 0
                consecutive_counts[bssid] = 0
                accumulated_rssi[bssid] = 0

            thr = rssi_threshold if thr_list is None else thr_list[timestep]
            if rssi >= thr:
                if timestep == consecutive_counts.get(bssid, 0):
                    consecutive_counts[bssid] += 1
                    accumulated_rssi[bssid] += rssi

        # print(consecutive_counts)

    # Finding the BSSID with the maximum consecutive count
    max_count = 0
    stable_bssids = []
    # print('Counts:\n')
    # print(consecutive_counts)

    for bssid, count in consecutive_counts.items():
        if count > max_count:
            max_count = count
            stable_bssids = [bssid]
        elif count == max_count:
            stable_bssids.append(bssid)

    return stable_bssids, consecutive_counts, accumulated_rssi


def format_preference_data(original_data_list, max_number_per_data=10):
    """Format a dataset for DPO where each row is a (prompt, chosen, rejected) tuple.

    :param max_number_per_data: max number of generated data for one original data.
    :param original_data_list: list of preprocessed data points from 'preprocess_v43'.
    :return: list of tuples used for DPO trainer.
    """
    formatted_data_list = []
    for data in original_data_list:
        prompt = data["messages"][:1]
        chosen_bssids = data["completion"]
        chosen, rejected = [], []
        chosen_cnt, rejected_cnt = 0, 0
        for bssid, cnt in data["counts"].items():
            if bssid in chosen_bssids and chosen_cnt < max_number_per_data:
                chosen.append([{"role": "assistant", "content": f"Chosen BSSID: {bssid}"}])
                chosen_cnt += 1
            elif bssid not in chosen_bssids and rejected_cnt < max_number_per_data:
                rejected.append([{"role": "assistant", "content": f"Chosen BSSID: {bssid}"}])
                rejected_cnt += 1

        n = min(len(chosen), len(rejected), max_number_per_data)
        pairs = list(zip(random.sample(chosen, n), random.sample(rejected, n)))
        for chosen_msg, rejected_msg in pairs:
            formatted_data_list.append(
                {
                    "prompt": copy.deepcopy(prompt),
                    "chosen": copy.deepcopy(chosen_msg),
                    "rejected": copy.deepcopy(rejected_msg),
                }
            )

    return formatted_data_list


def apply_dpo_chat_template(example, tokenizer):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        prompt_messages = example["prompt"]
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        # example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        # example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)

    return example


def step_handover(
    future_networks: list[dict],
    initial_bssid: str,
    threshold: int,
    num_handovers_weight: int = 10,
    stop_at_handover: bool = False,
):
    """
    Calculate the average RSSI, number of handovers, and connected BSSIDs within the future time steps for a given threshold and initial BSSID.
    :param future_networks: list of future networks as sorted (bssid, rssi) pairs
    :param initial_bssid: initial BSSID
    :param threshold: RSSI threshold
    :param num_handovers_weight: weight for number of handovers in the reward function
    :param stop_at_handover: whether to stop at the first handover
    :return: reward, total RSSI, number of handovers, connected BSSIDs, final BSSID
    """
    total_rssi = 0
    num_handovers = 0
    current_bssid = initial_bssid
    connected_bssids = []

    for networks in future_networks:
        connected_bssids.append(current_bssid)
        current_rssi = networks.get(current_bssid, -100)
        highest_bssid = max(networks.keys(), key=lambda x: networks[x])
        highest_rssi = networks[highest_bssid]
        total_rssi += current_rssi

        # Check if handover is needed
        if current_rssi < threshold and highest_rssi > current_rssi:
            current_bssid = highest_bssid
            num_handovers += 1
            if stop_at_handover:
                break

    # Calculate reward: average RSSI - average number of handovers
    reward = (total_rssi - num_handovers_weight * num_handovers) / len(connected_bssids)

    return reward, total_rssi, num_handovers, connected_bssids, current_bssid
