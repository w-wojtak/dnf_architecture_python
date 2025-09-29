# pylint: disable=C0200
from scipy.ndimage import gaussian_filter1d  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from src.utils import *
from pathlib import Path
import sys


def find_bump_centers(u_array, theta, bump_half_width=50):

    centers = []
    u_copy = u_array.copy()

    while np.max(u_copy) > theta:
        max_idx = np.argmax(u_copy)
        centers.append(max_idx)

        start = max(0, max_idx - bump_half_width)
        end = min(len(u_copy), max_idx + bump_half_width + 1)
        u_copy[start:end] = -1.5
        # print(centers)

    return centers


# Load most recent files
# data_dir = Path(__file__).resolve().parents[1] / "data"
# file1, file2 = find_latest_field_files(data_dir)

# u_field_1 = np.load(file1)
# u_field_2 = np.load(file2)

# # Create x-axis (based on the field shape)
# x = np.linspace(-80, 80, u_field_1.shape[0])


# # Plot final time step
# plt.figure(figsize=(10, 5))

# plt.subplot(2, 1, 1)
# plt.plot(x, u_field_1)
# plt.title("u_field_1 (final state)")
# plt.xlabel("x")
# plt.ylabel("activity")

# plt.subplot(2, 1, 2)
# plt.plot(x, u_field_2)
# plt.title("u_field_2 (final state)")
# plt.xlabel("x")
# plt.ylabel("activity")

# plt.tight_layout()
# plt.show()

fig = axs = line1_field = line1_input = line2_field = line2_input = line1_ud = None
line3_field = line4_field = line5_field = None


# beta_adapt = 0.0055
beta_adapt = 0.001

# Default to trial 1 if not provided
trial_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Define input_onset_time_2 values for different trials
input_onset_time_by_trial = {
    1: [8, 19, 32, 46, 58],   # Default
    2: [8, 19, 32, 46, 58],  # Example for trial 2
    3: [8, 19, 32, 46, 58],   # You can add more if needed
}

# Use the provided value if available, else fallback to default
input_onset_time_2 = input_onset_time_by_trial.get(
    trial_number, input_onset_time_by_trial[1])

plot_fields = True

plot_every = 5    # update plot every x time steps
plot_delay = 0.05   # delay (in seconds) before each plot update

# ====================================
# --------- Parameters ---------------
# ====================================

kernel_pars_act = [1.5, 0.8, 0.1]  # ok ADDED INHIBITION 0.5
kernel_pars_sim = [1.7, 0.8, 0.7]  # ok
kernel_pars_wm = [1.75, 0.5, 0.8]  # ok
# kernel_pars_wm = [2.0, 0.5, 0.8]  # ok
kernel_pars_f = [1.5, 0.8, 0.0]  # same as in act
kernel_pars_error = [1.5, 0.8, 0.0]  # same as in act

kernel_pars_inh = [3, 1.5, 0.0]

x_lim, t_lim = 80, 80
dx, dt = 0.05, 0.05
# theta = 1

# x = np.linspace(-x_lim, x_lim, 200)
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)


folder = "/home/wwojtak/dnf_architecture_python/data_basic"

file1, ts1 = find_latest_file_with_prefix(folder, "u_field_1_")
# file2, ts2 = find_latest_file_with_prefix(folder, "u_field_2_")
file3, ts3 = find_latest_file_with_prefix(folder, "u_d_")

# Optional: check if all timestamps match
if not (ts1 == ts3):
    raise ValueError("Timestamps do not match across all files.")

# Load data
u_field_1 = np.load(file1)
# u_field_2 = np.load(file2)
u_d = np.load(file3)

f_test = np.heaviside(u_field_1 - 1.5, 1)


# ====================================
# --------- Initialization -----------
# ====================================

# load fields from files to initialize

try:
    # Flatten and compute initial h from task duration
    u_d = u_d.flatten()
    h_d_initial = max(u_d)

    if trial_number == 1:
        # Use u_field_1 for u_act
        # u_act = u_field_1.flatten() - h_d_initial + 1.5
        u_act = u_field_1 - h_d_initial + 1.5
        input_action_onset = u_field_1.flatten()
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        # Use u_field_2 for u_sim
        u_sim = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_1.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

    else:
        # TODO: save and load last memory PLUS h_amem
        # now it's loading original memory so the timng is wrong
        data_dir = Path(os.getcwd()) / 'data_basic'
        data_dir = "/home/wwojtak/dnf_architecture_python/data_basic"
        print(f"Loading h_amem from {data_dir}")

        latest_h_amem_file, _ = find_latest_file_with_prefix(
            data_dir, "h_u_amem_")
        latest_h_amem = np.load(latest_h_amem_file, allow_pickle=True)

        # # Load saved current memory instead of u_field_1
        # print(f"Loading current memory from {data_dir}")
        # latest_memory_file, _ = find_latest_file_with_prefix(
        #     data_dir, "u_memory_")
        # saved_memory = np.load(latest_memory_file, allow_pickle=True)

        u_act = u_field_1.flatten() - h_d_initial + 1.5 - latest_h_amem
        input_action_onset = u_field_1.flatten() - latest_h_amem
        # u_act = saved_memory.flatten() - h_d_initial + 1.5 - latest_h_amem
        # input_action_onset = saved_memory.flatten() - latest_h_amem
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        # Use u_field_2 for u_sim (CHECK!!!!)
        u_sim = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_1.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    u_act = np.zeros(np.shape(x))
    h_u_act = -3.2 * np.ones(np.shape(x))


sim_centers = find_bump_centers(u_sim, -1.5)
x_centers = [round(x[idx]) for idx in sim_centers]
print(f"BUMP CENTERS: {x_centers}")


# plt.figure(figsize=(10, 4))
# plt.plot(u_sim, label='kernel_act: a=1.5, s=0.8, w=0.0')
# plt.title('Gaussian Kernels with Inhibition')
# plt.xlabel('x')
# plt.ylabel('Kernel value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
