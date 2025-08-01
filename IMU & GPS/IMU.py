#!/usr/bin/env python3

######################################################
# Copyright (c)2025 Han-Ieum Dream UP
# Author: HOHO
######################################################
# IMU.py
# This code reads data from the MPU9250/MPU9265 board
# (MPU6050 - accel/gyro, AK8963 - mag), calibrates the
# accelerometer, and integrates accel ?? velocity ?? displacement
# 
# with offset removal and Savitzky?Golay filtering.
#
######################################################

import math
import time
import sys
sys.path.append('../')
from mpu9250_i2c import mpu6050_conv, AK8963_conv
import numpy as np
from collections import deque
from scipy.signal import savgol_filter  # Savitzky Golay
from scipy.signal import savgol_coeffs

import logging
import threading

GRAVITY = 9.80665

class Kalman:
    def __init__(self, q_angle=0.001, q_gyro=0.003, r_measure=0.03):
        self.Q_angle = q_angle
        self.Q_gyro = q_gyro
        self.R_measure = r_measure

        self.angle = 0.0
        self.bias = 0.0

        self.P = [[0.0, 0.0], [0.0, 0.0]]

    def update(self, meas_angle, gyro_rate, dt):
        """Update Kalman filter with measurement and gyro rate"""

        self.angle += dt * (gyro_rate - self.bias)

        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_gyro * dt
        
        delta = (meas_angle - self.angle + 180.0) % 360 - 180.0
        y = delta

        S = self.P[0][0] + self.R_measure
        K0 = self.P[0][0] / S
        K1 = self.P[1][0] / S

        self.angle += K0 * y
        self.bias += K1 * y

        P00, P01, P10, P11 = self.P[0][0], self.P[0][1], self.P[1][0], self.P[1][1]
        
        self.P[0][0] = P00 - K0 * P00
        self.P[0][1] = P01 - K0 * P01
        self.P[1][0] = P10 - K1 * P00
        self.P[1][1] = P11 - K1 * P01

        self.angle = (self.angle + 360.0) % 360
        return self.angle

class IMU:	
    def __init__(self):
        self.accel_coeffs = [np.array([0.99814955, -0.13019077]), # accel calibration (ax, bx)
                             np.array([0.99804448, 0.2620614]),   # (ay, by)
                             np.array([0.98654515, 0.1548281])]  # (az, bz)
    
        self.gyro_coeffs  = np.array([0.3926849365234375,
                                     -2.9891204833984375,
                                      0.9511947631835938])
        self.mag_coeffs   = np.array([48.90632629394531,
                                      0.318145751953125,
                                     36.475467681884766])  # [??T]
        
        # SG filter window
        self._accel_window = deque(maxlen=51)
        
        # previous accel, vel for trapezoidal integration
        self._prev_accel = np.zeros(3)
        self._prev_vel   = np.zeros(3)

        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.accel_filtered = np.zeros(3)

        # only yaw Kalman
        self.kf_yaw = Kalman()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # LPF parameters
        self.fs   = 100.0
        self.fc   = 20.0
        self.dt   = 1.0 / self.fs
        self.RC      = 1.0 / (2 * math.pi * self.fc)
        self.alpha_l = self.dt / (self.RC + self.dt)
        self.accel_lpf = np.zeros(3)

        self.count = 0
        self.vel_body = np.zeros(3)
        self.pos_body = np.zeros(3)
        self.running = False
        self.thread = None
        self.accel = np.zeros(3)
        self._accel_zeroed_window = deque(maxlen=51)
    
    def read_sensors(self):
        ax, ay, az, gx, gy, gz = mpu6050_conv()
        mx, my, mz = AK8963_conv()

        a_raw = np.array([ax, ay, az])
        a_cal = np.zeros(3)
        for i, (m, b) in enumerate(self.accel_coeffs):
            a_cal[i] = m * a_raw[i] + b

        g_raw = np.array([gx, gy, gz])
        g_cal = g_raw - self.gyro_coeffs
        
        m_raw = np.array([mx, my, mz])
        m_cal = m_raw - self.mag_coeffs

        return a_cal, g_cal, m_cal
    
    def initialize_pose(self):
        acc, gyro, mag = self.read_sensors()
        ax, ay, az = acc
        gx, gy, gz = gyro
        mx, my, mz = mag
        
        # accel-based roll/pitch (for initial mag compensation)
        acc_roll  = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
        self.roll  = acc_roll
        self.pitch = acc_pitch
        
        self.yaw = self.compute_tilt_heading(mx, my, mz, acc_roll, acc_pitch)
        self.kf_yaw.angle = self.yaw

    def compute_tilt_heading(self, mx, my, mz, roll_deg, pitch_deg):
        roll_rad  = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)
        xh = mx * math.cos(pitch_rad) + mz * math.sin(pitch_rad)
        yh = (mx * math.sin(roll_rad) * math.sin(pitch_rad) +
              my * math.cos(roll_rad) -
              mz * math.sin(roll_rad) * math.cos(pitch_rad))
        heading = math.degrees(math.atan2(yh, xh))
        if heading < 0:
            heading += 360
        elif heading >= 360:
            heading -= 360
        return heading

    def body_rates_to_euler_rate(self, p, q, r, roll_deg, pitch_deg, yaw_deg, seq='zyx'):
        # unchanged
        R = math.radians(roll_deg)
        P = math.radians(pitch_deg)
        Y = math.radians(yaw_deg)
        sR, cR = math.sin(R), math.cos(R)
        sP, cP = math.sin(P), math.cos(P)
        if abs(cP) < 1e-2:
            cP = 1e-2 if cP >= 0 else -1e-2
        T = np.array([
            [1,       sR * math.tan(P),  cR * math.tan(P)],
            [0,               cR,              -sR      ],
            [0,      sR / cP,           cR / cP       ]
        ])
        omega = np.array([p, q, r], dtype=float)
        return tuple(np.linalg.inv(T).dot(omega))

    """
    Z??(Yaw)?? ?????? 2D ????(3??3) ???? ????.
    yaw=0????+X, 90????+Y, 180????-X, 270????-Y
    """
    def get_rotation_matrix(self, yaw_deg):
        psi = math.radians(yaw_deg)
        c, s = math.cos(psi), math.sin(psi)
        return np.array([
            [ c, -s, 0],
            [ s,  c, 0],
            [ 0,  0, 1],
        ])

    def zero_velocity_update(self, threshold=0.05):
        if len(self._accel_window) < self._accel_window.maxlen:
            return
        arr = np.vstack(self._accel_window)
        peak2peak = np.ptp(arr, axis=0)
        if np.all(peak2peak < threshold * GRAVITY):
            self.velocity[:]  = 0.0
            self._prev_vel[:] = 0.0

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        return False

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
    
    def get_data(self):
        return self.accel, self.yaw
    def get_yaw(self):
        return self.yaw

    def _update(self):
        SAMPLE_RATE = 100.0
        DT = 1.0 / SAMPLE_RATE
        # warm-up
        for _ in range(self._accel_window.maxlen):
            a_cal, g_cal, m_cal = self.read_sensors()
            rot_mat = self.get_rotation_matrix(self.yaw)
            a_w = rot_mat @ (a_cal * GRAVITY)
            a_ng = a_w - np.array([0,0,GRAVITY])
            self._accel_window.append(a_ng)
            time.sleep(DT)
        next_time = time.perf_counter()
        print("Starting real-time 100Hz update loop??")
        while self.running:
            next_time += DT
            a_cal, g_cal, m_cal = self.read_sensors()
            g_world = self.body_rates_to_euler_rate(*g_cal, self.roll, self.pitch, self.yaw)
            self.roll  += g_world[0] * DT
            self.pitch += g_world[1] * DT
            meas_yaw    = self.compute_tilt_heading(*m_cal, self.roll, self.pitch)
            self.yaw    = self.kf_yaw.update(meas_yaw, g_world[2], DT)

            # accel ?? world ?? non-gravity
            rot_mat = self.get_rotation_matrix(self.yaw)
            g_vec   = np.array([0,0,GRAVITY])
            a_w     = rot_mat @ (a_cal * GRAVITY)
            a_e     = a_w - g_vec
            a_ng    = np.array([a_e[1], a_e[0], -a_e[2]])

            self._accel_window.append(a_ng)
            drift   = np.mean(self._accel_window, axis=0)
            accel_z = a_ng - drift
            self._accel_zeroed_window.append(accel_z)
            if len(self._accel_zeroed_window) == self._accel_zeroed_window.maxlen:
                arr    = np.vstack(self._accel_zeroed_window)
                arr_sg = savgol_filter(arr, 51, 3, axis=0, mode='mirror')
                a_filt = arr_sg[-1]
            else:
                a_filt = accel_z
            self.accel_lpf = self.alpha_l * a_filt + (1 - self.alpha_l) * self.accel_lpf
            self.count += 1
            if self.count >= 200:
                self.zero_velocity_update()
                self.count = 0

            self.accel    = self.accel_lpf.copy()
            self.velocity = self._prev_vel + (self._prev_accel + self.accel)*(DT/2)
            self.position = self.position     + (self._prev_vel   + self.velocity)*(DT/2)
            self._prev_accel = self.accel.copy()
            self._prev_vel   = self.velocity.copy()

            # 7) body-frame velocity & position
            rot_mat    = self.get_rotation_matrix(self.yaw)
            R_w2b      = rot_mat.T
            vel_body   = R_w2b.dot(self.velocity)
            pos_body   = R_w2b.dot(self.position)
            self.vel_body = vel_body
            self.pos_body = pos_body

            now = time.perf_counter()
            sleep_t = next_time - now
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    imu = IMU()
    imu.initialize_pose()
    max_samples = 1000
    accel_raw_buf   = np.zeros((max_samples,3))
    accel_filt_buf  = np.zeros((max_samples,3))
    velocity_buf    = np.zeros((max_samples,3))
    position_buf    = np.zeros((max_samples,3))
    orientation_buf = np.zeros((max_samples,3))
    meas_yaw_buf    = np.zeros(max_samples)
    int_yaw_buf     = np.zeros(max_samples)
    timestamps      = np.zeros(max_samples)
    SAMPLE_RATE = 100.0
    DT = 1.0 / SAMPLE_RATE
    sample_count = 0
    samples_per_zvu = int(2 * SAMPLE_RATE)

    for _ in range(imu._accel_window.maxlen):
        a_cal, g_cal, m_cal = imu.read_sensors()
        rot_mat = imu.get_rotation_matrix(imu.yaw)
        a_w     = rot_mat @ (a_cal * GRAVITY)
        a_ng    = a_w - np.array([0,0,GRAVITY])
        imu._accel_window.append(a_ng)
        time.sleep(DT)
    next_time = time.perf_counter()
    yaw_int_prev = imu.yaw
    print("??? ???? ?????? ????. ???????? SG ?????? ???? ??????????.")
    print("Starting real-time 100Hz update loop with logging??")
    for i in range(max_samples):
        next_time += DT
        a_cal, g_cal, m_cal = imu.read_sensors()
        rot_mat = imu.get_rotation_matrix(imu.yaw)
        g       = np.array([0,0,GRAVITY])
        a_w     = rot_mat @ (a_cal * GRAVITY)
        a_e     = a_w - g
        a_ng    = np.array([a_e[1], a_e[0], -a_e[2]])
        imu._accel_window.append(a_ng)
        drift   = np.mean(np.vstack(imu._accel_window), axis=0)
        accel_z = a_ng - drift
        imu._accel_zeroed_window.append(accel_z)
        if len(imu._accel_zeroed_window) == imu._accel_zeroed_window.maxlen:
            arr    = np.vstack(imu._accel_zeroed_window)
            arr_sg = savgol_filter(arr, 51, 3, axis=0, mode='mirror')
            a_filt = arr_sg[-1]
        else:
            a_filt = accel_z
        imu.accel_lpf = imu.alpha_l * a_filt + (1 - imu.alpha_l) * imu.accel_lpf
        if (sample_count := sample_count + 1) >= samples_per_zvu:
            imu.zero_velocity_update()
            sample_count = 0
        vel = imu._prev_vel + (imu._prev_accel + imu.accel_lpf) * (DT/2)
        pos = imu.position    + (imu._prev_vel    + vel) * (DT/2)
        g_world = imu.body_rates_to_euler_rate(*g_cal, imu.roll, imu.pitch, imu.yaw)
        imu.roll  += g_world[0] * DT
        imu.pitch += g_world[1] * DT
        meas_yaw = imu.compute_tilt_heading(*m_cal, imu.roll, imu.pitch)
        yaw_int  = yaw_int_prev + g_world[2] * DT
        yaw_int_prev = yaw_int
        imu.yaw = imu.kf_yaw.update(meas_yaw, g_world[2], DT)
        accel_raw_buf[i]   = a_ng
        accel_filt_buf[i]  = imu.accel_lpf
        velocity_buf[i]    = vel
        position_buf[i]    = pos
        orientation_buf[i] = np.array([imu.roll, imu.pitch, imu.yaw])
        meas_yaw_buf[i]    = meas_yaw
        int_yaw_buf[i]     = yaw_int
        timestamps[i]      = i * DT
        imu._prev_accel = imu.accel_lpf.copy()
        imu._prev_vel   = vel.copy()
        imu.position    = pos.copy()
        now = time.perf_counter()
        sleep_t = next_time - now
        if sleep_t > 0:
            time.sleep(sleep_t)
    print(len(timestamps) / 10, "sampling rate")
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    for idx, label in zip([0,1,2], ['X','Y','Z']):
        axes[0].plot(timestamps, accel_filt_buf[:,idx], label=f'Accel_{label} (filtered)')
        axes[0].plot(timestamps, accel_raw_buf[:,idx], alpha=0.3, linestyle='--', label=f'Accel_{label} (raw)')
    axes[0].set_ylabel('Accel (m/s??)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    for idx, label in zip([0,1,2], ['X','Y','Z']):
        axes[1].plot(timestamps, velocity_buf[:,idx], label=f'Vel_{label}')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    for idx, label in zip([0,1,2], ['X','Y','Z']):
        axes[2].plot(timestamps, position_buf[:,idx], label=f'Pos_{label}')
    axes[2].set_ylabel('Displacement (m)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)
    axes[3].plot(timestamps, orientation_buf[:,0], label='Roll')
    axes[3].plot(timestamps, orientation_buf[:,1], label='Pitch')
    axes[3].plot(timestamps, orientation_buf[:,2], label='Yaw (KF)')
    axes[3].plot(timestamps, meas_yaw_buf, alpha=0.3, linestyle='--', label='Yaw_raw (mag)')
    axes[3].plot(timestamps, int_yaw_buf, alpha=0.3, linestyle=':', label='Yaw_int (gyro)')
    axes[3].set_ylabel('Angle (deg)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True)
    plt.tight_layout()
    plt.show()
