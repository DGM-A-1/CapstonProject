#!/usr/bin/env python3
######################################################
# Copyright (c) 2021 Maker Portal LLC
# Author: Joshua Hrisko (modified)
######################################################
#
# This code reads data from the MPU9250/MPU9265 board
# (MPU6050 - accel/gyro, AK8963 - mag), calibrates the
# accelerometer, and integrates accel ?? velocity ?? displacement
# with offset removal and Savitzky?Golay filtering.
#
######################################################

import time
import sys
sys.path.append('../')
from mpu9250_i2c import mpu6050_conv, AK8963_conv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter  # Savitzky?Golay ????

# Wait up to 5?s for IMU to connect
t0 = time.time()
start_bool = False
while time.time() - t0 < 5:
    try:
        _ = mpu6050_conv()
        start_bool = True
        break
    except:
        continue

time.sleep(2)  # wait for MPU to load and settle

# Number of samples to use for calibration
cal_size = 500

def accel_fit(x_input, m_x, b):
    """Linear model for accel calibration."""
    return m_x * x_input + b

def get_accel():
    """Read raw accel values (g) from MPU6050 via I2C."""
    ax, ay, az, _, _, _ = mpu6050_conv()
    return ax, ay, az

def accel_cal():
    """Calibrate accelerometer using +1g, ?1g, and 0g orientations per axis."""
    print("-" * 50)
    print("Accelerometer Calibration")
    mpu_offsets = [None, None, None]
    axis_vec = ['z', 'y', 'x']
    cal_dirs = ["upward", "downward", "perpendicular to gravity"]
    idxs = [2, 1, 0]

    for q, axis in enumerate(axis_vec):
        print("-" * 50)
        samples = [[], [], []]
        for i, direc in enumerate(cal_dirs):
            input(f"-------- Press Enter and keep IMU steady with {axis}-axis {direc}")
            for _ in range(cal_size):
                mpu6050_conv()
            vals = []
            while len(vals) < cal_size:
                try:
                    a = get_accel()
                    vals.append(a[idxs[q]])
                except:
                    continue
            samples[i] = np.array(vals)

        X = np.concatenate(samples)
        Y = np.concatenate([
            1.0 * np.ones_like(samples[0]),
           -1.0 * np.ones_like(samples[1]),
            0.0 * np.ones_like(samples[2])
        ])
        popt, _ = curve_fit(accel_fit, X, Y, maxfev=10000)
        mpu_offsets[idxs[q]] = popt

    print("Accelerometer Calibrations Complete:", mpu_offsets)
    return mpu_offsets

def imu_integrator():
    """Main loop: read calibrated accel, remove offset, apply SG filter, integrate, and plot."""
    data_index = 0    # 0=x, 1=y, 2=z
    dt_stop = 5       # record duration [s]

    plt.style.use('ggplot')
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    while True:
        t_array = []
        accel_raw = []
        print("Starting Data Acquisition")
        for ax in axs:
            ax.clear()

        t0 = time.time()
        while time.time() - t0 <= dt_stop:
            try:
                ax, ay, az, wx, wy, wz = mpu6050_conv()
                mx, my, mz = AK8963_conv()
                t = time.time() - t0
                raw_val = accel_fit(
                    [ax, ay, az, wx, wy, wz, mx, my, mz][data_index],
                    *accel_coeffs[data_index]
                )
                accel_raw.append(raw_val)
                t_array.append(t)
            except:
                continue
        print("Data Acquisition Stopped")

        # approximate sample rate
        fs = len(accel_raw) / dt_stop

        # 1) Offset removal
        accel_zeroed = np.array(accel_raw) - np.mean(accel_raw)

        # 2) Savitzky?Golay filter for smoothing
        #    window_length: odd integer <= len(accel_zeroed)
        #    polyorder: polynomial degree (< window_length)
        window_length = 51 if len(accel_zeroed) >= 51 else (len(accel_zeroed)//2)*2+1
        polyorder = 3
        accel_sg = savgol_filter(accel_zeroed, window_length, polyorder)

        # to m/s??
        accel_filt = accel_sg * 9.80665

        print(f"Sample Rate: {fs:.0f} Hz")
        # 3) Integrate ?? velocity ?? displacement
        veloc_array = np.concatenate([[0.0], cumtrapz(accel_filt, x=t_array)])
        dist_array  = np.concatenate([[0.0], cumtrapz(veloc_array, x=t_array)])

        print(f"Sample Rate: {fs:.0f}?Hz")
        print(f"Approximated displacement: {dist_array[-1]:.2f}?m")

        # Plot raw vs SG?filtered accel, velocity, displacement
        axs[0].plot(t_array, accel_raw, label="raw accel", alpha=0.3)
        axs[0].plot(t_array, accel_sg,   label="Savitzky?Golay accel", linewidth=2)
        axs[1].plot(t_array, veloc_array, label="velocity")
        axs[2].plot(t_array, dist_array,  label="displacement")

        axs[0].set_ylabel('Acceleration [g]', fontsize=14)
        axs[1].set_ylabel('Velocity [m/s]', fontsize=14)
        axs[2].set_ylabel('Displacement [m]', fontsize=14)
        axs[2].set_xlabel('Time [s]', fontsize=14)
        for ax in axs:
            ax.legend()
        axs[0].set_title("MPU9250 Accelerometer Integration", fontsize=16)

        plt.pause(0.01)
        plt.savefig("accel_veloc_displace_integration.png",
                    dpi=300, bbox_inches='tight', facecolor="#FCFCFC")

if __name__ == '__main__':
    if not start_bool:
        print("IMU not started ? check wiring and I2C settings.")
        sys.exit(1)

    # Calibrate accelerometer (or load old values)
    old_vals = True
    if not old_vals:
        accel_coeffs = accel_cal()
    else:
        accel_coeffs = [
            np.array([1.0096, -0.1838]),
            np.array([1.0003, -0.0815]),
            np.array([0.9840,  0.1561])
        ]

    # (Optional) quick check
    data = np.array([get_accel() for _ in range(cal_size)])

    # Start integration loop
    imu_integrator()
