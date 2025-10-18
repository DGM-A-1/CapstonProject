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
ANGLE_THRESHOLD  = 10.0    # 허용 방향 오차 범위 (degrees)
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
WHEEL_BASE       = 0.15    # meters

# Target geo-coordinate (lat, lon)
TARGET_LAT       = 35.8874055
TARGET_LON       = 128.611753

''' 두 개의 위도/경도 좌표 간의 방위각 <Yaw> (북쪽이 0도 , 시계방향으로 증가) 계산 '''
def bearing_deg(lat1, lon1, lat2, lon2):

    # 구면 삼각법을 사용한 방위각 계산 -> 지구는 둥글기 때문 /// 구면삼각법?? 공부
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2r) # 동서 방향 성분
    x = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon) # 남북 방향 성분
    brng = math.degrees(math.atan2(y, x)) # 최종 Yaw 계산

    return (brng + 360) % 360

''' 윤성이 부분 '''
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

''' 윤성이 부분 '''
def vw_to_pwm(v: float, w: float):
    """Convert linear v (m/s) and angular w (rad/s) to left/right PWM."""
    comp = w * ANGULAR_GAIN * WHEEL_BASE / 2.0
    vL = v - comp
    vR = v + comp
    return velocity_to_pwm(vL), velocity_to_pwm(vR)

''' 안쓰임 '''
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
    imu.start() # IMU에서 update() 쓰레드 시작
    print("IMU initialization complete")

    # Initialize GPS
    print("Initializing GPS...")
    gps = GPSReader()
    if not gps.start(): # GPSReader 에서 _read_loop 쓰레드 시작
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
            pos = gps.get_position() # (lat, lon)
            if pos is None:
                if loop_count % 10 == 0:
                    print("Waiting for GPS signal...")
                time.sleep(0.1)
                continue
            lat, lon = pos

            # 현재 위치와 목표 지점 사이의 남은거리(m)를 계산
            dist = geodesic((lat, lon), (TARGET_LAT, TARGET_LON)).meters

            # Change threshold when within 10m
            threshold = 30.0 if dist <= 10.0 else ANGLE_THRESHOLD

            # 목표 지점 도착 확인 (2m 이내)
            if dist <= 2.0:
                print(f"Distance to target: {dist:.2f} m")
                print("Successfully arrived target!")
                ser.write(b'P:0,0\n')
                break

            # 현재 위치에서 목표 지점까지의 방위각 계산
            target_bearing = bearing_deg(lat, lon, TARGET_LAT, TARGET_LON)
            # Yaw 차이 계산 (-180 ~ +180)
            angle_diff = (target_bearing - yaw + 540) % 360 - 180

            # 방향 오차가 임계값 이내일 때 전진, 아니면 회전
            if abs(angle_diff) <= threshold:
                mode = "Forward"
                left_pwm, right_pwm = vw_to_pwm(CRUISE_SPEED, 0.0)
                left_pwm = min(left_pwm + 10, MAX_PWM)
                right_pwm = max(right_pwm - 5, MIN_PWM)
            else:
                if angle_diff < 0:     # 목표 방향이 왼쪽에 있을 때
                    mode = "Turn Left"
                    left_pwm, right_pwm = 0, velocity_to_pwm(CRUISE_SPEED)
                else:                  # 목표 방향이 오른쪽에 있을 때
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
