# pylint: disable=C0200
import numpy as np
import matplotlib.pyplot as plt
# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter
from datetime import datetime
import os


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


# ====================================
# --------- Parameters ---------------
# ====================================

kernel_pars = [1, 0.7, 0.9]

x_lim, t_lim = 80, 100
dx, dt = 0.05, 0.05
theta = 1

tau_h = 20
h_0 = 0

input_flag = True
input_shape = [3, 1.5]   # same for both
input_duration = [1, 1, 1, 1, 1]  # same for both

# Positions for input set 1
input_position_1 = [-60, -30, 0, 30, 60]
input_onset_time_1 = [3, 8, 12, 16, 20]

# Positions for input set 2
input_position_2 = [-50, -30, 10, 35, 65]
input_onset_time_2 = [5, 10, 14, 18, 23]

# Pack parameters for each input set
input_pars_1 = [input_shape, input_position_1,
                input_onset_time_1, input_duration]
input_pars_2 = [input_shape, input_position_2,
                input_onset_time_2, input_duration]

save_video = False  # Set to True to save video of the simulation
plot_every = 5    # update plot every x time steps
plot_delay = 0.05   # delay (in seconds) before each plot update


# ====================================
# --------- Initialization -----------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

inputs_1 = get_inputs(x, t, dt, input_pars_1, input_flag)
inputs_2 = get_inputs(x, t, dt, input_pars_2, input_flag)

u_field_1 = h_0 * np.ones_like(x)
u_field_2 = h_0 * np.ones_like(x)

h_u_1 = h_0 * np.ones_like(x)
h_u_2 = h_0 * np.ones_like(x)

w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

history_u_1 = np.zeros((len(t), len(x)))
history_u_2 = np.zeros((len(t), len(x)))


# ====================================
# --------- Plot setup ---------------
# ====================================

plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)  # rows, col

# First subplot: field 1
line1_field, = axs[0].plot(x, u_field_1, label='Field activity u_field_1(x)')
line1_input, = axs[0].plot(x, inputs_1[0, :], label='Input 1')
axs[0].set_ylim(-2, 10)
axs[0].set_ylabel("Activity")
axs[0].legend()
axs[0].set_title("Field 1 - Time = 0")

# Second subplot: field 2
line2_field, = axs[1].plot(x, u_field_2, label='Field activity u_field_2(x)')
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

for i in range(len(t)):
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

    # Update plot every plot_every steps or at last step
    if i % plot_every == 0 or i == len(t) - 1:
        # Update field 1
        line1_field.set_ydata(u_field_1)
        line1_input.set_ydata(inputs_1[i, :])
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

# Save final field states
np.save(file_path_1, u_field_1)
np.save(file_path_2, u_field_2)

print(f"Saved u_field_1 to {file_path_1}")
print(f"Saved u_field_2 to {file_path_2}")

if save_video:
    writer.finish()

plt.ioff()
plt.show()
