import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_cov, accel_noise_var, gps_noise_lat_var, gps_noise_lon_var):
        """
        initial_state: np.array([x0, y0, vx0, vy0])
        initial_cov:    4x4 covariance matrix
        accel_noise_var: variance of acceleration noise (m^2/s^4)
        gps_noise_var:   variance of GPS position noise (m^2)
        """
        self.x = initial_state           # ???? ???? (4,)
        self.P = initial_cov             # ?????? (4x4)
        self.var_a = accel_noise_var     # ?????? ?????? ????
        self.R = np.diag([gps_noise_lat_var, gps_noise_lon_var])  # GPS ???? ???? ??????

    def predict(self, dt, accel_world, yaw_rad):
        # State transition
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ],
        ])
        
        # Control input: rotate body accel into world
        ax_w = accel_world[0]
        ay_w = accel_world[1]
        
        u = np.array([0.5*dt**2*ax_w, 0.5*dt**2*ay_w, dt*ax_w, dt*ay_w])
        
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