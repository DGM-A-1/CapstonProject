#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CSV ???????? ---
# numpy ???? (header=True, ???? ???? ???? ????)
data = np.genfromtxt('gps_coords.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
lats = data['latitude']
lons = data['longitude']

# --- 2. ?????? ???? ---
def compute_stats(arr, bins=50):
    mean = np.mean(arr)
    var  = np.var(arr)            # ddof=0, ?????? ????
    counts, edges = np.histogram(arr, bins=bins)
    idx = np.argmax(counts)
    mode = (edges[idx] + edges[idx+1]) / 2
    return mean, mode, var, counts, edges

lat_mean, lat_mode, lat_var, lat_counts, lat_edges = compute_stats(lats)
lon_mean, lon_mode, lon_var, lon_counts, lon_edges = compute_stats(lons)

# --- 3. ?????????? + ?????? ?????? ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Latitude plot
ax = axes[0]
ax.hist(lats, bins=lat_edges, alpha=0.7)
ax.set_title('Latitude Histogram')
ax.set_xlabel('Latitude')
ax.set_ylabel('Count')
# ?????? ??????
txt = (
    f"Mean: {lat_mean:.6f}\n"
    f"Mode: {lat_mode:.6f}\n"
    f"Variance: {lat_var:.6f}"
)
ax.text(0.95, 0.95, txt, transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', alpha=0.3))

# Longitude plot
ax = axes[1]
ax.hist(lons, bins=lon_edges, alpha=0.7)
ax.set_title('Longitude Histogram')
ax.set_xlabel('Longitude')
ax.set_ylabel('Count')
txt = (
    f"Mean: {lon_mean:.6f}\n"
    f"Mode: {lon_mode:.6f}\n"
    f"Variance: {lon_var:.6f}"
)
ax.text(0.95, 0.95, txt, transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', alpha=0.3))

plt.tight_layout()
plt.show()
