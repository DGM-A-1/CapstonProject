#!/usr/bin/env python3
"""
main.py
----------------------------------------------------------------
Sensor fusion of IMU and GPS using a linear Kalman Filter for 2D position & velocity.

- State vector: [x, y, vx, vy]
- Control input: acceleration in body frame, rotated to world frame.
- Measurements: GPS provides (x, y).

Added: real-time 2D trajectory plot showing INS, GPS, and KF estimates.
"""

import time
import numpy as np
import logging
import matplotlib.pyplot as plt

from IMU import IMU
from GPS import GPSReader


class KalmanFilter:
    def __init__(self, initial_state, initial_cov, accel_noise_var, gps_noise_var):
        """
        initial_state: np.array([x0, y0, vx0, vy0])
        initial_cov:    4x4 covariance matrix
        accel_noise_var: variance of acceleration noise (m^2/s^4)
        gps_noise_var:   variance of GPS position noise (m^2)
        """
        self.x = initial_state           # 상태 벡터 (4,)
        self.P = initial_cov             # 공분산 (4x4)
        self.var_a = accel_noise_var     # 가속도 노이즈 분산
        self.R = np.diag([gps_noise_var, gps_noise_var])  # GPS 측정 잡음 공분산

    def predict(self, dt, accel_body, yaw_rad):
        # State transition
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ],
        ])
        # Control input: rotate body accel into world
        ax_w = accel_body[0]*np.cos(yaw_rad) - accel_body[1]*np.sin(yaw_rad)
        ay_w = accel_body[0]*np.sin(yaw_rad) + accel_body[1]*np.cos(yaw_rad)
        u = np.array([0.5*dt**2*ax_w, 0.5*dt**2*ay_w, dt*ax_w, dt*ay_w])
        # Process noise
        q = self.var_a
        Q = np.array([
            [dt**4/4,      0, dt**3/2,      0],
            [0,      dt**4/4,      0, dt**3/2],
            [dt**3/2,      0,    dt**2,      0],
            [0,      dt**3/2,      0,    dt**2],
        ]) * q
        # Predict
        self.x = F @ self.x + u
        self.P = F @ self.P @ F.T + Q

    def update(self, z_pos):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z_pos - (H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Initialize IMU
    logging.info("Initializing IMU...")
    imu = IMU(bus_number=1)
    if not imu.start():
        logging.error("IMU initialization failed.")
        return

    # Initialize GPS
    logging.info("Initializing GPS...")
    gps = GPSReader()
    if not gps.start():
        logging.error("GPS failed to get a fix and establish an origin.")
        imu.stop()
        return

    # Initialize Kalman Filter
    init_pos = (0.0, 0.0)
    init_state = np.array([0.0, 0.0, 0.0, 0.0])
    init_cov = np.diag([1,1,1,1])
    kf = KalmanFilter(init_state, init_cov, accel_noise_var=0.1**2, gps_noise_var=2.0**2)

    # Prepare real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    ins_traj, gps_traj, kf_traj = [], [], []
    ins_line, = ax.plot([], [], 'r-', label='INS')
    gps_line, = ax.plot([], [], 'g-', label='GPS')
    kf_line, = ax.plot([], [], 'b-', label='KF')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend()

    last_time = time.time()
    last_gps = init_pos

    logging.info("Starting Kalman filter loop. Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            dt = now - last_time
            if dt <= 0:
                time.sleep(0.001)
                continue
            last_time = now

            # IMU prediction
            imu.update()
            ax_raw = imu.accel_filtered[0:2]  # non-gravity accel XY
            yaw = np.radians(imu.yaw)
            kf.predict(dt, ax_raw, yaw)

            # GPS update if new
            gps_pos = gps.get_position_xy_body(yaw_rad=)
            if gps_pos != last_gps:
                kf.update(np.array(gps_pos))
                last_gps = gps_pos

            # Collect trajectories
            ins_traj.append((imu.position[0], imu.position[1]))
            gps_traj.append(tuple(gps_pos))
            kf_traj.append((kf.x[0], kf.x[1]))

            # Update plot data
            ins_x, ins_y = zip(*ins_traj)
            gps_x, gps_y = zip(*gps_traj)
            kf_x, kf_y   = zip(*kf_traj)
            ins_line.set_data(ins_x, ins_y)
            gps_line.set_data(gps_x, gps_y)
            kf_line.set_data(kf_x, kf_y)

            # Rescale
            all_x = ins_x + gps_x + kf_x
            all_y = ins_y + gps_y + kf_y
            ax.set_xlim(min(all_x)-1, max(all_x)+1)
            ax.set_ylim(min(all_y)-1, max(all_y)+1)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Console output
            x, y, vx, vy = kf.x
            print(f"\rPos: [{x:+7.2f}, {y:+7.2f}] Vel: [{vx:+6.2f}, {vy:+6.2f}] Yaw: {imu.yaw:.1f}°", end='')

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        imu.stop()
        gps.stop()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()
