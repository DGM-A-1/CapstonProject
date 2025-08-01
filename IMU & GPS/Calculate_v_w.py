import math
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic


from IMU import IMU
from GPS import GPSReader
from KalmanFilter import KalmanFilter

# Calculate_V_W ?????? ?????? ?????? ?? ???? ?????? ????
class VelocityController:
    """???? ?????? ??????"""
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, max_speed=5.0, desired_dist=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed = max_speed
        self.desired_dist = desired_dist
        self.pid_integral = 0.0
        self.prev_error = 0.0
    
    
    """
    ?????????? ?????? ?????? ???? ?????? ???????? ?????? ??????????.
    
    Parameters:
    - current_state: np.array([x, y, vx, vy]) - ???????? ????
    - target_pos: (target_x, target_y) - ???? ????
    - current_heading_deg: ???? ???? [degrees]
    - dt: ???? ???? [s]
    
    Returns:
    - v: ?????? [m/s]
    - w: ?????? [rad/s]
    """
    def calculate_velocities(self, current_state, target_pos, current_heading_deg, dt):
        current_x, current_y = current_state[0], current_state[1]
        target_x, target_y = target_pos
        
        # ???? ????
        dx = target_x - current_x
        dy = target_y - current_y
        dist = math.hypot(dx, dy)
        
        # PID ????
        error = dist - self.desired_dist
        
        # ?????? ???? (Anti-windup)
        self.pid_integral += error * dt
        self.pid_integral = max(min(self.pid_integral, self.max_speed/self.ki), 
                               -self.max_speed/self.ki)
        
        # ?????? ????
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # PID ????
        output = self.kp*error + self.ki*self.pid_integral + self.kd*derivative
        v = max(0, min(output, self.max_speed))
        
        # ?????? ????
        los_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (los_angle - current_heading_deg) % 360
        angle_diff = angle_diff - 360 if angle_diff > 180 else angle_diff
        
        # ?????? (rad/s)
        angular_gain = 2.5
        w = math.radians(angular_gain * angle_diff * 0.05)
        
        # ???? ????????
        self.prev_error = error
        
        return v, w
    

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    geopy.geodesic?? ???? ?????????? ?? ???? NED (north, east) [m] ????
    """
    # ???? ????: ?????? ???????? ?????? ?????? ?????? ????
    north = geodesic((lat0, lon0), (lat, lon0)).meters
    # ???? ????: ?????? ???????? ?????? ?????? ?????? ????
    east  = geodesic((lat0, lon0), (lat0, lon)).meters

    # ?????? ?????? ????, ???????? ?????? ?????? ?????? ???? ????
    if lat < lat0:
        north *= -1
    if lon < lon0:
        east  *= -1

    return north, east


"""
???? ?????? ?????????? V(??????)?? W(??????)?? ???????? ???? ????

Parameters:
- target_x, target_y: ???? ???? [m]
"""
def Calculate_V_W_with_target(target_lat, target_lon):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 1)IMU ???? ???? ?? ???? ????
    logging.info("Initializing IMU...")
    imu = IMU()
    imu.initialize_pose()
    if not imu.start():
        logging.error("IMU failed to start.")
        return
    
    # 2)GPS ???? ???? ?? ???? ????
    logging.info("Initializing GPS...")
    gps = GPSReader()
    if not gps.start():
        logging.error("GPS failed to get a fix and establish an origin.")
        imu.stop()
        return
        
    # 2.75) lat lon?? NED X, Y?????? ????   
    origin_lat, origin_lon = gps.get_position()
    target_x, target_y = latlon_to_xy(
        target_lat, target_lon,
        origin_lat, origin_lon
    )
    
    
    # 3) ???????? ??????
    # ???? ???? = GPS?? ???? ????[N, E]?? ????
    init_pos = tuple(gps.get_position_xy())
    init_state = np.array([init_pos[0], init_pos[1], 0.0, 0.0])
    init_cov = np.diag([1,1,1,1])
    kf = KalmanFilter(init_state, init_cov, accel_noise_var=0.1**2, 
                      gps_noise_lat_var=3.426856, gps_noise_lon_var=1.433247)

    # 4) ???? ?????? ??????
    # PID ????????: KP, KI, KD, MAX_SPEED, DESIRED_DIST
    velocity_controller = VelocityController(kp=1.0, ki=0.1, kd=0.05, 
                                           max_speed=5.0, desired_dist=2.0)

    # 5) ?????? ???? ?????? => IMU / GPS / Kalman???? ?????? ????[N E] ?????? ????
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ins_traj, gps_traj, kf_traj = [], [], []
    ins_line, = ax1.plot([], [], 'r-', label='INS')
    gps_line, = ax1.plot([], [], 'g-', label='GPS')
    kf_line, = ax1.plot([], [], 'b-', label='KF')
    target_point, = ax1.plot(target_x, target_y, 'ko', markersize=10, label='Target')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    
    # Velocity plot
    v_history, w_history = [], []
    v_line, = ax2.plot([], [], 'b-', label='Linear Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Control Outputs')
    ax2.legend()

    # 6) ???? ????, ???? ??????
    last_time = time.time()
    start_time = time.time()
    last_gps = init_pos
    ############# ?????? ?? Setup ???? #####################

    # 7) ???? ???? ????
    logging.info("Starting Kalman filter loop with target tracking. Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            dt = now - last_time
            if dt <= 0:
                time.sleep(0.001)
                continue
            last_time = now

            # IMU prediction
            imu_accel, imu_yaw = imu.get_data() #accel = (NED frame) [m/s^2], yaw = degrees
            yaw = np.radians(imu_yaw) # Convert to radians
            kf.predict(dt, imu_accel, yaw)

            # GPS update if new
            gps_pos = gps.get_position_xy()
            if gps_pos != last_gps:
                kf.update(np.array(gps_pos))
                last_gps = gps_pos

            # Calculate control velocities
            v, w = velocity_controller.calculate_velocities(
                kf.x, (target_x, target_y), imu.yaw, dt
            )

            # Collect data
            ins_traj.append((imu.position[0], imu.position[1]))
            gps_traj.append(tuple(gps_pos))
            kf_traj.append((kf.x[0], kf.x[1]))
            v_history.append((now - start_time, v))
            w_history.append((now - start_time, w))

            # Update trajectory plot
            if len(ins_traj) > 0:
                ins_x, ins_y = zip(*ins_traj)
                gps_x, gps_y = zip(*gps_traj)
                kf_x, kf_y = zip(*kf_traj)
                ins_line.set_data(ins_x, ins_y)
                gps_line.set_data(gps_x, gps_y)
                kf_line.set_data(kf_x, kf_y)

                all_x = ins_x + gps_x + kf_x + (target_x,)
                all_y = ins_y + gps_y + kf_y + (target_y,)
                ax1.set_xlim(min(all_x)-1, max(all_x)+1)
                ax1.set_ylim(min(all_y)-1, max(all_y)+1)

            # Update velocity plot
            if len(v_history) > 0:
                t_vals, v_vals = zip(*v_history)
                ax2.clear()
                ax2.plot(t_vals, v_vals, 'b-', label='V (m/s)')
                ax2.plot(t_vals, [w for _, w in w_history], 'r-', label='W (rad/s)')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Velocity')
                ax2.set_title('Control Outputs')
                ax2.legend()
                ax2.grid(True)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Console output
            x, y, vx, vy = kf.x
            dist_to_target = math.hypot(target_x - x, target_y - y)
            print(f"\rPos: [{x:+7.2f}, {y:+7.2f}] Target Dist: {dist_to_target:6.2f}m "
                  f"V: {v:5.2f}m/s W: {w:+5.2f}rad/s", end='')

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        imu.stop()
        gps.stop()
        logging.info("Shutdown complete.")
        
        # Return final velocities
        return v, w
    
    
if __name__ == "__main__":
    # End point coordinate
    target_lat = 35.8874055
    target_lon = 128.611753
    Calculate_V_W_with_target(target_lat, target_lon)



'''
# ???????? ?????? V, W?? ???????? ????
def get_velocities_to_target(current_x, current_y, current_heading_deg, 
                            target_x, target_y, controller_state=None):
    """
    ???? ???????? ?????????? V, W?? ???????? ?????? ????
    
    Parameters:
    - current_x, current_y: ???? ???? [m]
    - current_heading_deg: ???? ???? [degrees]
    - target_x, target_y: ???? ???? [m]
    - controller_state: (pid_integral, prev_error) ???? ???? None
    
    Returns:
    - v: ?????? [m/s]
    - w: ?????? [rad/s]
    - new_controller_state: ?????????? (pid_integral, prev_error)
    """
    if controller_state is None:
        pid_integral, prev_error = 0.0, 0.0
    else:
        pid_integral, prev_error = controller_state
    
    v, w, new_integral, new_error = calculate_v_w_from_target(
        current_x, current_y, current_heading_deg,
        target_x, target_y, pid_integral, prev_error
    )
    
    return v, w, (new_integral, new_error)
    """???? ?????? ??????"""
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, max_speed=5.0, desired_dist=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed = max_speed
        self.desired_dist = desired_dist
        self.pid_integral = 0.0
        self.prev_error = 0.0
        
    def calculate_velocities(self, current_state, target_pos, current_heading_deg, dt):
        """
        ?????????? ?????? ?????? ???? ?????? ???????? ?????? ??????????.
        
        Parameters:
        - current_state: np.array([x, y, vx, vy]) - ???????? ????
        - target_pos: (target_x, target_y) - ???? ????
        - current_heading_deg: ???? ???? [degrees]
        - dt: ???? ???? [s]
        
        Returns:
        - v: ?????? [m/s]
        - w: ?????? [rad/s]
        """
        current_x, current_y = current_state[0], current_state[1]
        target_x, target_y = target_pos
        
        # ???? ????
        dx = target_x - current_x
        dy = target_y - current_y
        dist = math.hypot(dx, dy)
        
        # PID ????
        error = dist - self.desired_dist
        
        # ?????? ???? (Anti-windup)
        self.pid_integral += error * dt
        self.pid_integral = max(min(self.pid_integral, self.max_speed/self.ki), 
                               -self.max_speed/self.ki)
        
        # ?????? ????
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # PID ????
        output = self.kp*error + self.ki*self.pid_integral + self.kd*derivative
        v = max(0, min(output, self.max_speed))
        
        # ?????? ????
        los_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (los_angle - current_heading_deg) % 360
        angle_diff = angle_diff - 360 if angle_diff > 180 else angle_diff
        
        # ?????? (rad/s)
        angular_gain = 2.5
        w = math.radians(angular_gain * angle_diff * 0.05)
        
        # ???? ????????
        self.prev_error = error
        
        return v, w
'''