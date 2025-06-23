import os
import numpy as np
from datetime import datetime
from pathlib import Path


def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * (b * np.sin(abs(alpha * x)) + np.cos(alpha * x)))


def get_inputs(x, t, dt, input_pars, input_flag):
    if not input_flag:
        return np.zeros((len(t), len(x)))

    [input_shape, input_positions, input_onsets, input_durations] = input_pars
    amplitude, sigma = input_shape
    inputs = np.zeros((len(t), len(x)))

    for pos, onset, dur in zip(input_positions, input_onsets, input_durations):
        time_on = int(onset / dt)
        time_off = int((onset + dur) / dt)
        gaussian = amplitude * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        inputs[time_on:time_off, :] += gaussian

    return inputs


def find_latest_field_files(folder_path):
    folder_path = Path(folder_path)  # Ensure it's a Path object
    files = os.listdir(folder_path)
    print("All files found in data/:", files)

    field1_files = sorted(
        [f for f in files if f.startswith("u_field_1_") and f.endswith(".npy")])
    field2_files = sorted(
        [f for f in files if f.startswith("u_field_2_") and f.endswith(".npy")])
    print("u_field_1 files:", field1_files)
    print("u_field_2 files:", field2_files)

    if not field1_files or not field2_files:
        raise FileNotFoundError(
            "No valid matching timestamped field files found.")

    # Extract timestamps
    def extract_timestamp(filename):
        # Assumes format: u_field_X_YYYYMMDD_HHMMSS.npy
        return "_".join(filename.split("_")[-2:]).replace(".npy", "")

    timestamps1 = [extract_timestamp(f) for f in field1_files]
    timestamps2 = [extract_timestamp(f) for f in field2_files]
    common_ts = sorted(set(timestamps1) & set(timestamps2))

    if not common_ts:
        raise FileNotFoundError(
            "No matching timestamps found between field1 and field2 files.")

    latest_ts = common_ts[-1]
    file1 = folder_path / f"u_field_1_{latest_ts}.npy"
    file2 = folder_path / f"u_field_2_{latest_ts}.npy"

    return file1, file2
