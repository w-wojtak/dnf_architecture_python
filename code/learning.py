import numpy as np
import matplotlib.pyplot as plt

# --------- Kernel function ---------


def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * ((b * np.sin(abs(alpha * x))) + np.cos(alpha * x)))

# --------- Input signal generator ---------


def get_inputs(x, t, dt, input_pars, input_flag):
    if not input_flag:
        return np.zeros((len(t), len(x)))

    [input_shape, input_positions, input_onsets, input_durations] = input_pars
    amplitude, sigma = input_shape
    inputs = np.zeros((len(t), len(x)))

    for pos, onset, dur in zip(input_positions, input_onsets, input_durations):
        center_idx = np.argmin(np.abs(x - pos))
        time_on = int(onset / dt)
        time_off = int((onset + dur) / dt)
        gaussian = amplitude * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        inputs[time_on:time_off, :] += gaussian

    return inputs

# --------- Plotting ---------


def plot_final_state_1d(history_u, field_pars):
    x_lim, t_lim, dx, dt, _ = field_pars
    x = np.arange(-x_lim, x_lim + dx, dx)
    plt.figure(figsize=(8, 4))
    plt.plot(x, history_u[-1, :])
    plt.xlabel("Space")
    plt.ylabel("Field activity")
    plt.title(f"Final state at t = {t_lim}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_space_time_flat(history_u, field_pars):
    x_lim, t_lim, dx, dt, _ = field_pars
    x = np.arange(-x_lim, x_lim + dx, dx)
    t = np.arange(0, t_lim + dt, dt)
    plt.figure(figsize=(10, 5))
    plt.imshow(history_u, aspect='auto', extent=[
               x[0], x[-1], t[-1], t[0]], cmap='hot')
    plt.colorbar(label='Activity')
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Space-Time Activity Map")
    plt.tight_layout()
    plt.show()

# ====================================
# --------- Parameters --------------
# ====================================


# Kernel parameters
kernel_pars = [1, 0.7, 0.9]

# Field parameters
x_lim, t_lim = 80, 100
dx, dt = 0.05, 0.05
theta = 1
field_pars = [x_lim, t_lim, dx, dt, theta]

# Adaptation parameters
tau_h = 20
h_0 = 0

# Input parameters
input_flag = True
input_shape = [3, 1.5]
input_position = [-60, -30, 0, 30, 60]
input_onset_time = [3, 8, 12, 16, 20]
input_duration = [1, 1, 1, 1, 1]
input_pars = [input_shape, input_position, input_onset_time, input_duration]

# ====================================
# --------- Initialization -----------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)
history_u = np.zeros((len(t), len(x)))

inputs = get_inputs(x, t, dt, input_pars, input_flag)

u_field = np.ones_like(x) * h_0
h_u = np.ones_like(x) * h_0

# Precompute FFT of kernel
w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

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

# ====================================
# --------- Plotting -----------------
# ====================================

plot_final_state_1d(history_u, field_pars)
# plot_space_time_flat(history_u, field_pars)

# Optional: Save data
np.save("data/sequence_memory.npy", history_u)
np.save("data/field_parameters.npy", field_pars)
