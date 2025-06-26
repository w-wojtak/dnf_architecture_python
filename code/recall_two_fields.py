# pylint: disable=C0200
import numpy as np
import matplotlib.pyplot as plt
# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter
from datetime import datetime
import os
from src.utils import *
from pathlib import Path

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

trial_number = 1


# ====================================
# --------- Parameters ---------------
# ====================================

kernel_pars_act = [1.5, 0.8, 0.0]  # ok
kernel_pars_sim = [1.7, 0.8, 0.7]  # ok
kernel_pars_wm = [1.75, 0.5, 0.8]  # ok
kernel_pars_f = [1.5, 0.8, 0.0]  # same as in act
kernel_pars_error = [1.5, 0.8, 0.0]  # same as in act

x_lim, t_lim = 80, 100
dx, dt = 0.05, 0.05
theta = 1

# x = np.linspace(-x_lim, x_lim, 200)
x = np.arange(-x_lim, x_lim + dx, dx)

# Compute kernels
kernel_act = kernel_gauss(x, *kernel_pars_act)
kernel_sim = kernel_gauss(x, *kernel_pars_sim)

# Compute FFTs
w_hat_act = np.fft.fft(kernel_act)
w_hat_sim = np.fft.fft(kernel_sim)

# # Plot kernels
# plt.figure(figsize=(10, 4))
# plt.plot(x, kernel_act, label='kernel_act: a=1.5, s=0.8, w=0.0')
# plt.plot(x, kernel_sim, label='kernel_sim: a=1.7, s=0.8, w=0.7', linestyle='--')
# plt.title('Gaussian Kernels with Inhibition')
# plt.xlabel('x')
# plt.ylabel('Kernel value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

folder = "data"

file1, ts1 = find_latest_file_with_prefix(folder, "u_field_1_")
file2, ts2 = find_latest_file_with_prefix(folder, "u_field_2_")
file3, ts3 = find_latest_file_with_prefix(folder, "u_d_")

# Optional: check if all timestamps match
if not (ts1 == ts2 == ts3):
    raise ValueError("Timestamps do not match across all files.")

# Load data
u_field_1 = np.load(file1)
u_field_2 = np.load(file2)
u_d = np.load(file3)


# plt.figure(figsize=(10, 4))
# plt.plot(x, u_field_1, label='u 1')
# plt.plot(x, u_d, label='u d', linestyle='--')
# plt.xlabel('x')
# plt.ylabel(' value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# load fields from files to initialize

try:
    # Flatten and compute initial h from task duration
    u_d = u_d.flatten()
    h_d_initial = max(u_d)

    if trial_number == 1:
        # Use u_field_2 for u_act
        u_act = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset = u_field_1.flatten()
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        # Use u_field_1 for u_sim
        u_sim = u_field_2.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_2.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

    else:
        data_dir = Path(os.getcwd()) / 'dnf_architecture_extended/data'
        print(f"Loading h_amem from {data_dir}")

        latest_h_amem_file, _ = find_latest_file_with_prefix(
            data_dir, "h_amem_")
        latest_h_amem = np.load(latest_h_amem_file, allow_pickle=True)

        u_act = u_field_1.flatten() - h_d_initial + 1.5 + latest_h_amem
        input_action_onset = u_field_1.flatten() + latest_h_amem
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    u_act = np.zeros(np.shape(x))
    h_u_act = -3.2 * np.ones(np.shape(x))
