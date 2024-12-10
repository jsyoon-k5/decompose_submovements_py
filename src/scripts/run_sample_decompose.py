import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.signal import savgol_filter

from ..decomposer.decompose_subm import decompose_submovements
from ..utils.mymath import derivative_central
from ..utils.myplot import figure_grid, figure_save

datapath = os.path.join(Path(__file__).parent.parent.parent, "data/sample/raw.csv")
data = pd.read_csv(datapath)

timestamp = data["time"].to_numpy() # Second unit
pos_x = data["pos_x"].to_numpy()    # Meter unit
pos_y = data["pos_y"].to_numpy()    # Meter unit

# Velocity
vel_x = derivative_central(timestamp, pos_x)
vel_y = derivative_central(timestamp, pos_y)

# Filtering
time_interval = np.mean(np.diff(timestamp))
vel_x = savgol_filter(vel_x, round(0.151 / time_interval), 5)
vel_y = savgol_filter(vel_y, round(0.151 / time_interval), 5)

# Downsample
new_timestamp = np.arange(0, timestamp[-1], 0.01)
vel_x = np.interp(new_timestamp, timestamp, vel_x)
vel_y = np.interp(new_timestamp, timestamp, vel_y)
vel = np.vstack((vel_x, vel_y)).T

# Decompose
movements, parameters, segment = decompose_submovements(
    hand_vel=vel,
    time_interval=0.01,
    verbose=True,
    save_path=os.path.join(Path(__file__).parent.parent.parent, "data/sample/decompose")
)

# Visualize
fig, axs = figure_grid(2, 1, size_ax=np.array([6, 3]))

for r in range(2):
    ax = axs[r]
    ax.plot(movements["timestamp"], movements[f"orig_v{'xy'[r]}"], color='k', linewidth=0.5, label='Original')
    ax.plot(movements["timestamp"], movements[f"recn_v{'xy'[r]}"], color='r', linewidth=0.5, label='Reconstructed')
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Velocity ({'xy'[r]}) (m/s)")
    ax.set_xlim(0, movements["timestamp"].to_numpy()[-1])
    ax.set_ylim(-0.21, 0.21)
    ax.legend()

figure_save(fig, os.path.join(Path(__file__).parent.parent.parent, "data/sample/decompose/result.png"))