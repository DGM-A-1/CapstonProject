# Pure_GPS.py (?????? ????)
# --------------------------------------------------------------------------
# GPS ???? ???? ?? ?????? ???? - NE ?????? ?? PID ???? ????
# --------------------------------------------------------------------------

import threading
import time
import logging
import math
import serial
import numpy as np
from geopy.distance import geodesic
from IMU import IMU
from GPS import GPSReader

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 2025-07-29 : ros ????
import rospy
from geometry_msgs.msg import Point

class VelocityController:
    """
    ???? ?????? ???????????????????? ?????? ????.
    ???? ?????????? ???????? ???????? ???????? ????.
    """
    def __init__(self,
                 kp=1.0, ki=0.1, kd=0.05,
                 max_speed=1.0, desired_dist=2.0,
                 target_lat=35.8874055, target_lon=128.611753,
                 serial_port='/dev/ttyUSB0', baudrate=115200):
        # PID params
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_speed = max_speed
        self.desired_dist = desired_dist
        self.pid_integral = 0.0
        self.prev_error = 0.0

        # ???? ???? (lat/lon)
        self.target_lat, self.target_lon = target_lat, target_lon

        # Serial (console ??????)
        self.serial_port, self.baudrate = serial_port, baudrate

        # state variables
        self.v = 0.0
        self.w = 0.0
        self.running = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)

        # ???? ?? ????
        self.imu = None
        self.gps = None

        # ???????? ?????? ????
        self.gps_traj = []   # GPS ????
        self.history_t = []
        self.history_v = []
        self.history_w = []
        
        # ROS publisher
        self.loc_pub = None
        
        self.wheel_base = 0.15
        self.MIN_PWM = 240
        self.MAX_PWM = 250
        
        self.pwm_left = 0
        self.pwm_right = 0

    def _latlon_to_xy(self, lat, lon, lat0, lon0):
        """NE ??????: +x=????, +y=????"""
        north = geodesic((lat0, lon0), (lat, lon0)).meters
        east  = geodesic((lat0, lon0), (lat0, lon)).meters
        if lat < lat0:  north *= -1
        if lon < lon0:  east  *= -1
        return north, east  # x=north, y=east

    def _compute_pid(self, dist, dt):
        error = dist - self.desired_dist
        self.pid_integral += error * dt
        # anti-windup
        self.pid_integral = max(min(self.pid_integral, self.max_speed/self.ki),
                               -self.max_speed/self.ki)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp*error + self.ki*self.pid_integral + self.kd*derivative
        self.prev_error = error
        return max(0.0, min(output, self.max_speed))

    def _compute_angular(self, angle_diff):
        """PID Control.py ?????? ?????? ????"""
        angular_gain = 2.5
        return math.radians(angular_gain * angle_diff * 0.05)

    def bearing_deg(self, dx, dy):
        """
        NE ?????????? ???? 0??, ???? 90?? ???????? bearing ????
        dx: ???? ???? ???? (+x)
        dy: ???? ???? ???? (+y)
        """
        # atan2(????, ????) = atan2(y, x)
        brng = math.degrees(math.atan2(dy, dx))
        # ???? 0?? ???? ???????????? ????
        bearing = (90 - brng) % 360
        return bearing

    def normalize_angle_diff(self, target_angle, current_angle):
        """
        PID Control.py?? ???? ???? ???? ???? ????
        -180?? ~ +180?? ?????? ??????
        """
        angle_diff = (target_angle - current_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        return angle_diff

    def start(self):
        logging.info("Starting VelocityController...")
        
        #????) --- ROS ???? ?????? & ???????? ???? ---
        rospy.init_node('velocity_controller', anonymous=True)
        self.loc_pub = rospy.Publisher('my_location', Point, queue_size=10)
        
        # IMU init
        self.imu = IMU()
        self.imu.initialize_pose()
        if not self.imu.start():
            logging.error("IMU failed to start.")
            return False
        
        # GPS init
        self.gps = GPSReader()
        if not self.gps.start():
            logging.error("GPS failed to start.")
            self.imu.stop()
            return False
        
        # Thread ????
        self.running = True
        self.thread.start()
        return True

    def speed_to_pwm(self, vel):
        ratio = vel / self.max_speed
        pwm = int(round(255 * ratio))
        return max(self.MIN_PWM, min(pwm, self.MAX_PWM))

    def _run_loop(self):
        last_time = time.time()
        origin_lat, origin_lon = self.gps.get_position()
        
        target_x, target_y = self._latlon_to_xy(
            self.target_lat, self.target_lon,
            origin_lat, origin_lon
        )
        start_time = last_time
    
        while self.running and not rospy.is_shutdown():
            now = time.time()
            dt = now - last_time
            if dt <= 0:
                time.sleep(0.001)
                continue
            last_time = now

            # 1) sensor ?????? ????
            x, y = self.gps.get_position_xy()
            imu_yaw = self.imu.get_yaw()
            
            # 2) ?????? ????
            dx, dy = target_x - x, target_y - y  # dx=????????, dy=????????
            dist = math.hypot(dx, dy)

            # ???? ???? ???? (NE ??????)
            target_bearing = self.bearing_deg(dx, dy)

            print(f"Current Yaw: {imu_yaw:.2f}??, Target Yaw: {target_bearing:.2f}??")
            print(f"Distance to target: {dist:.2f}m")
            
            # ???? ?????? ?????????? ????
            if dist <= self.desired_dist:
                print("Goal Reached")
                self.v = 0.0
                self.w = 0.0
                self.pwm_left = 0
                self.pwm_right = 0
                self.running = False
                break   # while ???? ???????? cleanup???? ????

            # PID Control.py ???????? ???? ???? ????
            angle_diff = self.normalize_angle_diff(target_bearing, imu_yaw)

            # ???? ?????? ?????? ???? (?????? 10??)
            if abs(angle_diff) > 10:
                print(f"Adjusting heading: angle_diff = {angle_diff:.2f}??")
                self.v = 0.0  # ???? ?????? ???? ???? 0
                self.w = self._compute_angular(angle_diff)
            else:
                # ?????? ?????? ????
                print("Heading aligned, moving forward")
                self.v = self._compute_pid(dist, dt)
                self.w = 0.0
                
            # 3) pwm ????
            half_wb = self.wheel_base / 2.0
            
            v_left = self.v - self.w * half_wb
            v_right = self.v + self.w * half_wb

            if(v_left != 0.0 or v_right != 0.0):
                self.pwm_left  = self.speed_to_pwm(abs(v_left)) * (1 if v_left >= 0 else -1)
                self.pwm_right = self.speed_to_pwm(abs(v_right)) * (1 if v_right >= 0 else -1)
            else:
                self.pwm_left = 0
                self.pwm_right = 0
          
            # --- ROS?? ???? ???? ???? ---
            if self.loc_pub is not None:
                msg = Point()
                msg.x = float(x)
                msg.y = float(y)
                msg.z = 0.0
                self.loc_pub.publish(msg)
            
            # 4) ???????? ?????? ????
            self.gps_traj.append((x, y))
            t_rel = now - start_time
            self.history_t.append(t_rel)
            self.history_v.append(self.v)
            self.history_w.append(self.w)

            time.sleep(0.1)

        # Cleanup
        self.imu.stop()
        self.gps.stop()
        logging.info("VelocityController loop terminated.")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        logging.info("VelocityController stopped.")


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    # 1) ???????? ????
    controller = VelocityController()
    if not controller.start():
        print("Failed to start controller.")
        return

    # --- origin?? target_xy ???? ------------------
    origin_lat, origin_lon = controller.gps.get_position()
    target_x, target_y = controller._latlon_to_xy(
    controller.target_lat, controller.target_lon,
    origin_lat, origin_lon
    )
    # ---------------------------------------------

    # 2) ?????? (???? ??????)
    ser = serial.Serial(controller.serial_port,
                        controller.baudrate, timeout=1)
    time.sleep(2)

    # 3) matplotlib ????
    fig, (ax_traj, ax_vel) = plt.subplots(1, 2, figsize=(12, 5))
    line_gps, = ax_traj.plot([], [], 'g-', label='GPS')
    target_point, = ax_traj.plot(target_x, target_y, 'ko', markersize=10, label='Target')

    ax_traj.set_xlabel('X (m) - North'); ax_traj.set_ylabel('Y (m) - East')
    ax_traj.set_title('Trajectory (NE Coordinate)'); ax_traj.legend()
    line_v, = ax_vel.plot([], [], 'b-', label='V (m/s)')
    line_w, = ax_vel.plot([], [], 'r-', label='W (rad/s)')
    ax_vel.set_xlabel('Time (s)'); ax_vel.set_ylabel('Velocity')
    ax_vel.set_title('Control Outputs'); ax_vel.legend()
    ax_vel.grid(True)

    # 4) ???? ???? ????
    def update(frame):
        # ??????????
        if controller.gps_traj:
            gps_x, gps_y = zip(*controller.gps_traj)

            line_gps.set_data(gps_x, gps_y)

            all_x = gps_x +  (target_x,)
            all_y =  gps_y + (target_y,)
            ax_traj.set_xlim(min(all_x)-1, max(all_x)+1)
            ax_traj.set_ylim(min(all_y)-1, max(all_y)+1)

        # ???? ????
        if controller.history_t:
            line_v.set_data(controller.history_t, controller.history_v)
            line_w.set_data(controller.history_t, controller.history_w)
            ax_vel.set_xlim(0, max(controller.history_t)+0.1)
            min_vw = min(min(controller.history_v), min(controller.history_w))
            max_vw = max(max(controller.history_v), max(controller.history_w))
            ax_vel.set_ylim(min_vw-0.1, max_vw+0.1)

        # ???????? ???? ?????? ????
        cmd = f"P:{int(controller.pwm_left)},{int(controller.pwm_right)}\n"
        ser.write(cmd.encode('utf-8'))
        return line_gps, line_v, line_w

    # 5) ?????????? ???? (100ms????)
    ani = FuncAnimation(fig, update, interval=100, blit=False)

    # 6) ???? & ???? ????
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Stopping...")
        ser.write(b"P:0,0\n")  # Stop command
        pass
    finally:
        controller.stop()
        ser.close()

if __name__ == '__main__':
    main()