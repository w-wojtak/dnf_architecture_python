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

u_field_1 = np.load(file1)
u_field_2 = np.load(file2)

# Create x-axis (based on the field shape)
x = np.linspace(-80, 80, u_field_1.shape[0])


# Plot final time step
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(x, u_field_1)
plt.title("u_field_1 (final state)")
plt.xlabel("x")
plt.ylabel("activity")

plt.subplot(2, 1, 2)
plt.plot(x, u_field_2)
plt.title("u_field_2 (final state)")
plt.xlabel("x")
plt.ylabel("activity")

plt.tight_layout()
plt.show()
