#!/usr/bin/env python3
"""
Main control script with debugging output
Adds distance reporting, dynamic angle threshold, and arrival detection.
Debug print statements combined for concise output.
"""
import math
import time
import serial
import sys
from geopy.distance import geodesic

from IMU import IMU
from GPS import GPSReader

# ----- Configuration -----
SERIAL_PORT      = '/dev/ttyUSB0'  # Arduino serial port
BAUDRATE         = 115200

# Motion parameters
ANGLE_THRESHOLD  = 10.0    # deg tolerance for straight
ALIGN_SPEED      = 30.0    # deg/s rotation speed
CRUISE_SPEED     = 0.3     # m/s forward speed

# PWM calibration (must match Arduino)
MAX_PWM          = 250
MIN_PWM          = 240
CALIB_MIN_V      = 1.0 / 4.2
CALIB_MAX_V      = 1.0 / 2.7
PWM_SLOPE        = (MAX_PWM - MIN_PWM) / (CALIB_MAX_V - CALIB_MIN_V)
PWM_INTERCEPT    = MIN_PWM - PWM_SLOPE * CALIB_MIN_V

# Robot parameters
ANGULAR_GAIN     = 1.0
WHEEL_BASE       = 0.15     # meters

# Target geo-coordinate (lat, lon)
TARGET_LAT       = 35.8874055
TARGET_LON       = 128.611753

# ----- Helper functions -----

def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Compute bearing from point1 to point2 in degrees (0=N, clockwise).
    """
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360


def velocity_to_pwm(v: float) -> int:
    """Map velocity (m/s) to PWM (0..255), forward-only."""
    if abs(v) < 0.01:
        return 0
    a = abs(v)
    if a <= CALIB_MIN_V:
        pwm = MIN_PWM
    elif a >= CALIB_MAX_V:
        pwm = MAX_PWM
    else:
        pwm = int(PWM_SLOPE * a + PWM_INTERCEPT + 0.5)
    return max(min(pwm, MAX_PWM), MIN_PWM)


def vw_to_pwm(v: float, w: float):
    """Convert linear v (m/s) and angular w (rad/s) to left/right PWM."""
    comp = w * ANGULAR_GAIN * WHEEL_BASE / 2.0
    vL = v - comp
    vR = v + comp
    return velocity_to_pwm(vL), velocity_to_pwm(vR)


def get_direction_name(angle):
    """Convert angle to compass direction name"""
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((angle+22.5)/45)%8]

# ----- Main -----
def main():
    print("=== Smart Swarm Driving System Starting ===")
    print(f"Target position: {TARGET_LAT}, {TARGET_LON}")
    print(f"Angle threshold: ±{ANGLE_THRESHOLD}° (default)")
    print(f"Forward speed: {CRUISE_SPEED} m/s")
    print(f"Rotation speed: {ALIGN_SPEED} deg/s")
    print("-" * 50)

    # Initialize IMU
    print("Initializing IMU...")
    imu = IMU()
    imu.initialize_pose()
    imu.start()
    print("IMU initialization complete")

    # Initialize GPS
    print("Initializing GPS...")
    gps = GPSReader()
    if not gps.start():
        print("GPS initialization failed!")
        return
    print("GPS initialization complete")

    # Serial to Arduino
    print(f"Connecting to Arduino ({SERIAL_PORT})...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        time.sleep(2)
        print("Arduino connection established")
    except Exception as e:
        print(f"Arduino connection failed: {e}")
        return

    print("\nStarting control...\n")
    loop_count = 0
    try:
        while True:
            loop_count += 1
            # Get sensor data
            yaw = imu.get_yaw()  # deg
            pos = gps.get_position()
            if pos is None:
                if loop_count % 10 == 0:
                    print("Waiting for GPS signal...")
                time.sleep(0.1)
                continue
            lat, lon = pos

            # Distance to target
            dist = geodesic((lat, lon), (TARGET_LAT, TARGET_LON)).meters

            # Change threshold when within 10m
            threshold = 30.0 if dist <= 10.0 else ANGLE_THRESHOLD

            # Arrival condition
            if dist <= 2.0:
                print(f"Distance to target: {dist:.2f} m")
                print("Successfully arrived target!")
                ser.write(b'P:0,0\n')
                break

            # Compute desired bearing and difference
            target_bearing = bearing_deg(lat, lon, TARGET_LAT, TARGET_LON)
            angle_diff = (target_bearing - yaw + 540) % 360 - 180

            # Decide action
            if abs(angle_diff) <= threshold:
                mode = "Forward"
                left_pwm, right_pwm = vw_to_pwm(CRUISE_SPEED, 0.0)
                left_pwm = min(left_pwm + 10, MAX_PWM)
                right_pwm = max(right_pwm - 5, MIN_PWM)
            else:
                if angle_diff < 0:
                    mode = "Turn Left"
                    left_pwm, right_pwm = 0, velocity_to_pwm(CRUISE_SPEED)
                else:
                    mode = "Turn Right"
                    left_pwm, right_pwm = velocity_to_pwm(CRUISE_SPEED), 0

            # Combined debug print every 0.5s
            if loop_count % 10 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Mode={mode} | "
                    f"Dist={dist:.2f}m | "
                    f"Yaw={yaw:.1f}°->Target={target_bearing:.1f}° (Diff={angle_diff:+.1f}°) | "
                    f"Thresh={threshold}° | PWM L={left_pwm:>3d}, R={right_pwm:>3d}"
                )
                print("-" * 40)

            # Send command
            cmd = f'P:{left_pwm},{right_pwm}\n'
            ser.write(cmd.encode('ascii'))
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        imu.stop()
        gps.stop()
        ser.write(b'P:0,0\n')
        ser.close()
        print("System shutdown complete")

if __name__ == '__main__':
    main()
