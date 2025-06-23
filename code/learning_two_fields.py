import numpy as np
import matplotlib.pyplot as plt
# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter


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
input_shape = [3, 1.5]
input_position = [-60, -30, 0, 30, 60]
input_onset_time = [3, 8, 12, 16, 20]
input_duration = [1, 1, 1, 1, 1]
input_pars = [input_shape, input_position, input_onset_time, input_duration]

save_video = False  # Set to True to save video of the simulation
plot_every = 5    # update plot every 100 time steps
plot_delay = 0.05   # delay (in seconds) before each plot update


# ====================================
# --------- Initialization -----------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

inputs = get_inputs(x, t, dt, input_pars, input_flag)

u_field = h_0 * np.ones_like(x)
h_u = h_0 * np.ones_like(x)

w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

history_u = np.zeros((len(t), len(x)))


# ====================================
# --------- Plot setup ---------------
# ====================================

plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))

line1, = ax.plot(x, u_field, label='Field activity u(x)')
line2, = ax.plot(x, inputs[0, :], label='Input')

ax.set_ylim(-2, 10)
ax.set_xlabel("x")
ax.set_ylabel("activity")
ax.legend()
ax.set_title("Time = 0")

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
    f = np.heaviside(u_field - theta, 1)
    f_hat = np.fft.fft(f)
    conv = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))
    h_u += dt / tau_h * f
    u_field += dt * (-u_field + conv + inputs[i, :] + h_u)
    history_u[i, :] = u_field

    # Update plot every plot_every steps or at last step
    if i % plot_every == 0 or i == len(t) - 1:
        line1.set_ydata(u_field)
        line2.set_ydata(inputs[i, :])
        ax.set_title(f"Time = {t[i]:.2f}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Pause to create delay before next update
        plt.pause(plot_delay)

        if save_video:
            writer.grab_frame()

if save_video:
    writer.finish()

plt.ioff()
plt.show()
