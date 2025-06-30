# pylint: disable=C0200
import numpy as np
import matplotlib.pyplot as plt
# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter
from datetime import datetime
import os
from src.utils import *


# ====================================
# --------- Parameters ---------------
# ====================================


plot_fields = False
save_video = False  # Set to True to save video of the simulation
# Predefine to avoid linter warnings
fig = axs = line1_field = line1_input = line2_field = line2_input = line1_ud = None

kernel_pars = [1, 0.7, 0.9]

x_lim, t_lim = 80, 50
dx, dt = 0.1, 0.1
theta = 1

tau_h = 20
h_0 = 0

h_0_d = 0
tau_h_d = 20
# theta_d = 1.5
# kernel_pars_d = [1, 0.7, 0.9]  # SAME as kernel_pars

input_flag = True
input_shape = [3, 1.5]   # same for both
input_duration = [1, 1, 1, 1, 1]  # same for both

# Positions for input set 1
input_position_1 = [-60, -30, 0, 30, 60]
input_onset_time_1 = [13, 18, 22, 26, 30]

# Positions for input set 2
input_position_2 = input_position_1  # [-50, -30, 10, 35, 65]
input_onset_time_2 = [15, 20, 24, 28, 33]

# Pack parameters for each input set
input_pars_1 = [input_shape, input_position_1,
                input_onset_time_1, input_duration]
input_pars_2 = [input_shape, input_position_2,
                input_onset_time_2, input_duration]


plot_every = 5    # update plot every x time steps
plot_delay = 0.05   # delay (in seconds) before each plot update


# ====================================
# --------- Initialization -----------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

input_indices_1 = [np.argmin(np.abs(x - pos)) for pos in input_position_1]

inputs_1 = get_inputs(x, t, dt, input_pars_1, input_flag)
inputs_2 = get_inputs(x, t, dt, input_pars_2, input_flag)

u_field_1 = h_0 * np.ones_like(x)
u_field_2 = h_0 * np.ones_like(x)

h_u_1 = h_0 * np.ones_like(x)
h_u_2 = h_0 * np.ones_like(x)

u_d = h_0_d * np.ones_like(x)
h_u_d = h_0_d * np.ones_like(x)

w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

history_u_1 = np.zeros((len(t), len(x)))
history_u_2 = np.zeros((len(t), len(x)))
# u_d_history = []  # List to store values at each time step
history_u_d = np.zeros((len(t), len(x)))

# ====================================
# --------- Plot setup ---------------
# ====================================

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)  # rows, col

    # First subplot: field 1 + u_d
    line1_field, = axs[0].plot(
        x, u_field_1, label='Field activity u_field_1(x)')
    line1_input, = axs[0].plot(x, inputs_1[0, :], label='Input 1')
    line1_ud, = axs[0].plot(
        x, u_d, label='Task duration field u_d(x)', linestyle='--')

    axs[0].set_ylim(-2, 10)
    axs[0].set_ylabel("Activity")
    axs[0].legend()
    axs[0].set_title("Field 1 - Time = 0")

    # Second subplot: field 2
    line2_field, = axs[1].plot(
        x, u_field_2, label='Field activity u_field_2(x)')
    line2_input, = axs[1].plot(x, inputs_2[0, :], label='Input 2')
    axs[1].set_ylim(-2, 10)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Activity")
    axs[1].legend()
    axs[1].set_title("Field 2 - Time = 0")


# Video writer setup
if save_video:
    metadata = dict(title='Field evolution', artist='Matplotlib',
                    comment='Field simulation video')
    writer = FFMpegWriter(fps=30, metadata=metadata)
    writer.setup(fig, "field_simulation.mp4", dpi=100)

# ====================================
# --------- Simulation loop ----------
# ====================================

# time_counter = 0.0
u_1_tc = []
u_2_tc = []
u_d_tc = []

for i in range(len(t)):

    # Input at time t=0 for the task duration field
    # if 0.0 <= time_counter < 1.0:
    if 0 <= i < 1/dt:
        input_d = 3.0 * np.exp(-((x - 0) ** 2) / (2 * 1.5 ** 2))
    else:
        input_d = 0.0

    f_1 = np.heaviside(u_field_1 - theta, 1)
    f_hat_1 = np.fft.fft(f_1)
    conv_1 = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_1 * w_hat)))
    h_u_1 += dt / tau_h * f_1
    u_field_1 += dt * (-u_field_1 + conv_1 + inputs_1[i, :] + h_u_1)
    history_u_1[i, :] = u_field_1

    f_2 = np.heaviside(u_field_2 - theta, 1)
    f_hat_2 = np.fft.fft(f_2)
    conv_2 = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_2 * w_hat)))
    h_u_2 += dt / tau_h * f_2
    u_field_2 += dt * (-u_field_2 + conv_2 + inputs_2[i, :] + h_u_2)
    history_u_2[i, :] = u_field_2

    f_d = np.heaviside(u_d - theta, 1)
    f_hat_d = np.fft.fft(f_d)
    conv_d = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_d * w_hat)))
    h_u_d += dt / tau_h_d * f_d
    u_d += dt * (-u_d + conv_d + input_d + h_u_d)
    history_u_d[i, :] = u_d

    u_1_values = [u_field_1[idx] for idx in input_indices_1]
    u_2_values = [u_field_2[idx] for idx in input_indices_1]
    u_d_value = u_d[int(len(x) / 2)]

    u_1_tc.append(u_1_values)
    u_2_tc.append(u_2_values)
    u_d_tc.append(u_d_value)

    # time_counter += dt

    # Update plot every plot_every steps or at last step
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        # Update field 1
        line1_field.set_ydata(u_field_1)
        line1_input.set_ydata(inputs_1[i, :])
        line1_ud.set_ydata(u_d)  # <-- update u_d line
        axs[0].set_title(f"Field 1 - Time = {t[i]:.2f}")

        # Update field 2
        line2_field.set_ydata(u_field_2)
        line2_input.set_ydata(inputs_2[i, :])
        axs[1].set_title(f"Field 2 - Time = {t[i]:.2f}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(plot_delay)

        if save_video:
            writer.grab_frame()


print(f"Max of u_sm1: {max(u_field_1)}")
print(f"Max of u_sm2: {max(u_field_2)}")
print(f"Max of u_d: {max(u_d)}")

# ====================================
# --------- Save Final Fields --------
# ====================================

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Get current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define file paths
file_path_1 = f"data/u_field_1_{timestamp}.npy"
file_path_2 = f"data/u_field_2_{timestamp}.npy"
file_path_3 = f"data/u_d_{timestamp}.npy"

# Save final field states
np.save(file_path_1, u_field_1)
np.save(file_path_2, u_field_2)
np.save(file_path_3, u_d)

print(f"Saved u_field_1 to {file_path_1}")
print(f"Saved u_field_2 to {file_path_2}")
print(f"Saved u_d to {file_path_3}")

if save_video:
    writer.finish()

plt.ioff()
plt.show()


u_f1_history = np.array(u_1_tc)
u_f2_history = np.array(u_2_tc)
u_d_history = np.array(u_d_tc)

timesteps = np.arange(len(u_f1_history))

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

# Plot u_field_1
for i, pos in enumerate(input_position_1):
    axs[0].plot(timesteps, u_f1_history[:, i], label=f'x = {pos}')
axs[0].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[0].set_ylabel('u_field_1')
axs[0].set_ylim(-1, 5)
axs[0].legend()
axs[0].grid(True)

# Plot u_field_2
for i, pos in enumerate(input_position_1):
    axs[1].plot(timesteps, u_f2_history[:, i], label=f'x = {pos}')
axs[1].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[1].set_ylabel('u_field_2')
axs[1].set_xlabel('Timestep')
axs[1].set_ylim(-1, 5)
axs[1].legend()
axs[1].grid(True)

# Plot u_d
# for i, pos in enumerate([0]):
axs[2].plot(timesteps, u_d_history, label='x = 0')
axs[2].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[2].set_ylabel('u_d')
axs[2].set_xlabel('Timestep')
axs[2].set_ylim(-1, 3)
axs[2].set_xlim(0, 100)
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
