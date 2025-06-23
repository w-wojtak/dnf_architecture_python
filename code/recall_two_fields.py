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
data_dir = Path(__file__).resolve().parents[1] / "data"
file1, file2 = find_latest_field_files(data_dir)

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
