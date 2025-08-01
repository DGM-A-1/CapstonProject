#!/usr/bin/env python3
"""
Enhanced IMU Motion Tracking for Raspberry Pi with 3D Visualization
Converted from Arduino code for MPU6050/MPU9250 with AK8963 magnetometer
"""

import smbus
import time
import math
import numpy as np
import csv
import logging
import matplotlib.pyplot as plt
import threading
from collections import deque
from datetime import datetime
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

# I2C 주소 및 레지스터 주소
MPU_ADDR = 0x68        # MPU6050/MPU9250 I2C address
MAG_ADDR = 0x0C        # AK8963 magnetometer I2C address
AK8963_CNTL = 0x0A     # Magnetometer control register
AK8963_XOUT_L = 0x03   # Magnetometer data start register
AK8963_ST2 = 0x09 # 데이터가 정상인지 검증하기 위함

# Constants
READ_INTERVAL = 0.01   # 센서 읽기·상태 업데이트를 100 Hz로 샘플링
GRAVITY = 9.81         # 중력가속도

# Enhanced thresholds for better motion detection
ACCEL_ALPHA = 0.8           # 노이즈 필터를 위한 계수 --> 원하는 반응 속도와 노이즈 제거 수준의 균형을 위한 계수
ACCEL_THRESHOLD = 0.01      # 저역 필터 후 비중력 가속도가 이 이하일 때 0으로 간주하기 위한 계수
VEL_THRESHOLD = 0.05        # m/s - 속도를 0으로 리셋하기위한 계수

STATIONARY_THRESHOLD = 0.15 # m/s - 정지상태를 감지하기 위한 임계치
VEL_DECAY_FACTOR = 0.85     # 연속 정지 모드에서 속도를 강제로 줄여 줄 때 곱해 주는 감쇠 계수
STATIONARY_COUNT_LIMIT = 20 # 정지상태를 판정하기 위한 카운트

""" 칼만필터 클래스 """ 
class Kalman:

    # 상태변수와 공분산
    # Q는 프로세스 잡음 공분산 행렬(모델의 불확실 정도를 나타냄) -> 튜닝
    # R은 측정 잡음 공분산 행렬(센서의 불확실 정도를 나타냄) -> 튜닝
    def __init__(self, q_angle=0.001, q_gyro=0.003, r_measure=0.03):
        self.Q_angle = q_angle
        self.Q_gyro = q_gyro
        self.R_measure = r_measure

        self.angle = 0.0 # 추정된 각도
        self.bias = 0.0 # 자이로 바이어스 추정치

        # P는 실시간으로 변하는 오차 공분산 행렬
        self.P = [[0.0, 0.0], [0.0, 0.0]]

    def update(self, meas_angle, gyro_rate, dt):
        """Update Kalman filter with measurement and gyro rate"""

        # Predict 단계
        self.angle += dt * (gyro_rate - self.bias) # 각도 예측

        # 공분산 행렬 P 갱신
        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle) 
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_gyro * dt
        
        # Update 단계
        # 칼만 이득 계산 -> 이전 공분산 행렬(P)을 사용함
        S = self.P[0][0] + self.R_measure
        K0 = self.P[0][0] / S
        K1 = self.P[1][0] / S

        # (측정된 각도) − (예측 각도)
        y = meas_angle - self.angle

        # 추정값 계산 -> 예측값과 측정값을 이용해 추정값을 도출함
        self.angle += K0 * y
        self.bias += K1 * y

        # 공분산 행렬 보정
        self.P[0][0] -= K0 * self.P[0][0]
        self.P[0][1] -= K0 * self.P[0][1]
        self.P[1][0] -= K1 * self.P[0][0]
        self.P[1][1] -= K1 * self.P[0][1]
        
        return self.angle

""" IMU 클래스 """
class IMU:
    def __init__(self, bus_number=1):

        # I2C 버스 열기
        self.bus = smbus.SMBus(bus_number)
        
        # 센서 별로 오프셋값 저장
        self.gyro_offset = [0.0, 0.0, 0.0]
        self.mag_offset = [0.0, 0.0, 0.0]
        self.accel_offset = [0.0, 0.0, 0.0]
        
        # Roll,Pitch,Yaw에 대한 Kalman filters 생성
        self.kf_roll = Kalman()
        self.kf_pitch = Kalman()
        self.kf_yaw = Kalman()
        
        # 현재 위치를 알기 위한 변수들
        self.velocity = [0.0, 0.0, 0.0]  # 속도 (m/s)
        self.position = [0.0, 0.0, 0.0]  # 위치 (m)  ----> 초기위치를 0,0,0으로 그냥 둬도 되나??
        self.accel_filtered = [0.0, 0.0, 0.0]  # 비중력가속도 (m/s²)
        self.stationary_count = 0             # 연속 정지 판정 카운터
        
        # 현재 추정 각도
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Rasberry Pi에서 그래프를 보기위한 변수들
        # Data storage for visualization
        self.max_points = 1000
        self.position_history = deque(maxlen=self.max_points)
        self.velocity_history = deque(maxlen=self.max_points)
        self.angle_history = deque(maxlen=self.max_points)
        
        self.last_time = time.time()
        self.running = False 

    """ 초기자세 측정 """
    def initialize_pose(self):
        ax, ay, az, gx, gy, gz, mx, my, mz = self.read_sensors()
        acc_roll = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))

        self.kf_roll.angle = acc_roll
        self.kf_pitch.angle = acc_pitch

        self.roll = acc_roll
        self.pitch = acc_pitch

    """ Write a byte to I2C device """
    def write_byte(self, addr, reg, val):
        self.bus.write_byte_data(addr, reg, val)

    """ Read multiple bytes from I2C device """    
    def read_bytes(self, addr, reg, length):
        return self.bus.read_i2c_block_data(addr, reg, length)
    
    """ Read 16-bit signed integer from I2C device """
    def read_i16(self, addr, reg):
        data = self.bus.read_i2c_block_data(addr, reg, 2)
        value = (data[0] << 8) | data[1]
        return value if value < 32768 else value - 65536
    
    """ IMU센서 초기화 함수 """
    def init_mpu(self):
        print("Initializing MPU...")
        try:
            self.write_byte(MPU_ADDR, 0x6B, 0x00)  # Wake up
            time.sleep(0.1)
            
            self.write_byte(MPU_ADDR, 0x37, 0x02)
            time.sleep(0.1)


            self.write_byte(MPU_ADDR, 0x1A, 0x03)  # Low pass filter ~44Hz
            self.write_byte(MPU_ADDR, 0x1B, 0x08)  # Gyro range 500deg/s
            self.write_byte(MPU_ADDR, 0x1C, 0x10)  # Accel range 8g
            print("MPU initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing MPU: {e}")
            return False
    
    """ 정지상태에서 자이로센서의 오프셋 측정 """
    def calibrate_gyro(self, samples=1000):
        
        print("Calibrating gyroscope... Keep sensor still!")
        sx = sy = sz = 0
        
        for i in range(samples):
            try:
                sx += self.read_i16(MPU_ADDR, 0x43)
                sy += self.read_i16(MPU_ADDR, 0x45)
                sz += self.read_i16(MPU_ADDR, 0x47)
                if i % 100 == 0:
                    print(".", end="", flush=True)
                time.sleep(0.005)
            except Exception as e:
                print(f"Error during gyro calibration: {e}")
                return False
        
        self.gyro_offset[0] = sx / samples
        self.gyro_offset[1] = sy / samples
        self.gyro_offset[2] = sz / samples
        print(" Done!")
        return True
    
    """ 정지상태에서 가속도센서의 오프셋 측정 """ 
    def calibrate_accel(self, samples=1000):
    
        print("Calibrating accelerometer... Keep sensor still and level!")
        sx = sy = sz = 0
        
        for i in range(samples):
            try:
                sx += self.read_i16(MPU_ADDR, 0x3B)
                sy += self.read_i16(MPU_ADDR, 0x3D)
                sz += self.read_i16(MPU_ADDR, 0x3F)
                if i % 100 == 0:
                    print(".", end="", flush=True)
                time.sleep(0.005)
            except Exception as e:
                print(f"Error during accel calibration: {e}")
                return False
        
        self.accel_offset[0] = sx / samples
        self.accel_offset[1] = sy / samples
        self.accel_offset[2] = sz / samples - 4096.0  # Remove 1g from Z-axis
        print(" Done!")
        print(f"Accel offsets: {self.accel_offset[0]:.1f}, {self.accel_offset[1]:.1f}, {self.accel_offset[2]:.1f}")
        return True
    
    """ 지자계센서의 오프셋 측정 """  
    def calibrate_mag(self, samples=200):
        
        print("Calibrating magnetometer... Rotate sensor in all directions!")
        sx = sy = sz = 0
        
        for i in range(samples):
            try:
                mx, my, mz = self.read_raw_mag()
                sx += mx
                sy += my
                sz += mz
                if i % 20 == 0:
                    print(".", end="", flush=True)
                time.sleep(0.025)
            except Exception as e:
                print(f"Error during mag calibration: {e}")
                return False
        
        self.mag_offset[0] = sx / samples
        self.mag_offset[1] = sy / samples
        self.mag_offset[2] = sz / samples
        print(" Done!")
        return True
     
    """ 지자계센서 데이터값 측정 """
    def read_raw_mag(self):
        
        try:
            self.write_byte(MAG_ADDR, AK8963_CNTL, 0x01) #  AK8963_CNTL에 0x01을 써서 지자계가 한 번 데이터를 측정
            time.sleep(0.01)
            data = self.read_bytes(MAG_ADDR, AK8963_XOUT_L, 6) # 6바이트 연속으로 읽어옴
            
            x = (data[1] << 8) | data[0]
            y = (data[3] << 8) | data[2]
            z = (data[5] << 8) | data[4]
            
            # Convert to signed
            x = x if x < 32768 else x - 65536
            y = y if y < 32768 else y - 65536
            z = z if z < 32768 else z - 65536
            
            # 물리 단위(µT) 환산
            mx = x * (1200.0 / 4096.0)
            my = y * (1200.0 / 4096.0)
            mz = z * (1200.0 / 4096.0)
            
            return mx, my, mz
        except Exception as e:
            print(f"Error reading magnetometer: {e}")
            return 0.0, 0.0, 0.0

    """ IMU가 기울어진 상태에서도 정확한 방위각을 얻기 위한 함수  """ 
    def compute_tilt_heading(self, mx, my, mz, roll_deg, pitch_deg):
        
        roll_rad = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)
        
        xh = mx * math.cos(pitch_rad) + mz * math.sin(pitch_rad)
        yh = (mx * math.sin(roll_rad) * math.sin(pitch_rad) + 
              my * math.cos(roll_rad) - 
              mz * math.sin(roll_rad) * math.cos(pitch_rad))
        
        heading = math.degrees(math.atan2(yh, xh))
        return heading + 360 if heading < 0 else heading
    
    ''' 가속도계 & 자이로 & 지자계 센서들의 raw 데이터를 모두 계산함 '''
    def read_sensors(self):

        try:
            # Read accelerometer (??8g range, 4096 LSB/g)
            ax = (self.read_i16(MPU_ADDR, 0x3B) - self.accel_offset[0]) / 4096.0
            ay = (self.read_i16(MPU_ADDR, 0x3D) - self.accel_offset[1]) / 4096.0
            az = (self.read_i16(MPU_ADDR, 0x3F) - self.accel_offset[2]) / 4096.0
            
            # Read gyroscope (??500??/s range, 65.5 LSB/??/s)
            gx = (self.read_i16(MPU_ADDR, 0x43) - self.gyro_offset[0]) / 65.5
            gy = (self.read_i16(MPU_ADDR, 0x45) - self.gyro_offset[1]) / 65.5
            gz = (self.read_i16(MPU_ADDR, 0x47) - self.gyro_offset[2]) / 65.5
            
            # Read magnetometer
            mx, my, mz = self.read_raw_mag()
            mx -= self.mag_offset[0]
            my -= self.mag_offset[1]
            mz -= self.mag_offset[2]
            
            return ax, ay, az, gx, gy, gz, mx, my, mz
        
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    """ 순수 비중력가속도 계산 """
    def calculate_non_gravity_accel(self, ax, ay, az, roll_deg, pitch_deg, dt):
        
        roll_rad = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)
        
        # roll, pitch를 이용해 중력 벡터 계산
        grav_x = -math.sin(pitch_rad)
        grav_y = math.sin(roll_rad) * math.cos(pitch_rad)
        grav_z = math.cos(roll_rad) * math.cos(pitch_rad)
        
        # 중력 성분 제거 및 단위 변환
        raw_non_grav = [
            (ax - grav_x) * GRAVITY,
            (ay - grav_y) * GRAVITY,
            (az - grav_z) * GRAVITY
        ]
        
        # 가속도계 관련 ==> 노이즈를 제거하기 위한 1차 LPF 구현, ACCEL_ALPHA 값이 작아지면 이전상태를 좀 더 신뢰함. -> 튜닝필요함
        non_grav_accel = [0, 0, 0]
        for i in range(3):
            self.accel_filtered[i] = (ACCEL_ALPHA * self.accel_filtered[i] + 
                                    (1.0 - ACCEL_ALPHA) * raw_non_grav[i])
            
            '''# 임계치 적용하여 드리프트를 줄임
            if abs(self.accel_filtered[i]) < ACCEL_THRESHOLD:
                self.accel_filtered[i] = 0
            non_grav_accel[i] = self.accel_filtered[i]'''
        
        return non_grav_accel # 실제로 움직이는 가속도 값

    """ 비중력 가속도 → 속도 → 위치를 계산 """   
    def update_motion(self, non_grav_accel, dt):
        
        # 정지 상태 판정
        total_accel = math.sqrt(sum(a*a for a in non_grav_accel))
        accel_threshold = STATIONARY_THRESHOLD
        vel_threshold = VEL_THRESHOLD
        
        # Check if device is stationary
        is_stationary = (total_accel < accel_threshold and 
                    all(abs(v) < vel_threshold for v in self.velocity))
        
        if is_stationary:
            self.stationary_count += 1
        else:
            self.stationary_count = 0
        
        # 속도 업데이트 & 드리프트 보정 ## 왜 강제로 감쇠시키는지 모르겟음 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!
        for i in range(3):
            if self.stationary_count > STATIONARY_COUNT_LIMIT:
                # 충분히 정지 상태가 유지됐으면 강제 감쇠
                self.velocity[i] *= VEL_DECAY_FACTOR
                if abs(self.velocity[i]) < VEL_THRESHOLD:
                    self.velocity[i] = 0
            elif is_stationary:
                # 아직 카운트 미달이지만 정지 중이니 천천히 감쇠
                self.velocity[i] *= 0.98
                if abs(self.velocity[i]) < VEL_THRESHOLD:
                    self.velocity[i] = 0
            else:
                # 이동 중일 때 가속도를 적분 -> 속도
                self.velocity[i] += non_grav_accel[i] * dt
                # 아주 미세한 장기 드리프트 방지용 바이어스 보정
                self.velocity[i] *= 0.9995
            
            # 위치 업데이트
            self.position[i] += self.velocity[i] * dt
        
        return total_accel, is_stationary
    
    ''' 센서 측정값 + 칼만필터로 정확한 Roll,Pitch,Yaw 값을 도출함 '''
    def update_angles(self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
        
        # 가속도계로부터 Roll,Pitch를 계산 --> 칼만필터에서 측정치 의미
        acc_roll = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
        
        # Kalman 필터 업데이트 (자이로 + 가속도)
        self.roll = self.kf_roll.update(acc_roll, gx, dt)
        self.pitch = self.kf_pitch.update(acc_pitch, gy, dt)
        
        # 자력계 + Tilt 보정으로 Yaw 측정치 계산
        mag_yaw = self.compute_tilt_heading(mx, my, mz, self.roll, self.pitch)

        # Kalman 필터 업데이트 (자이로 + 자력계)
        self.yaw = self.kf_yaw.update(mag_yaw, gz, dt)
        
        # Normalize yaw to 0-360 range
        while self.yaw < 0:
            self.yaw += 360
        while self.yaw >= 360:
            self.yaw -= 360
    
    """ IMU센서의 메인 함수 """
    def update(self):
        
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt < READ_INTERVAL:
            return False
        
        self.last_time = current_time
        
        # 센서 데이터 읽기
        ax, ay, az, gx, gy, gz, mx, my, mz = self.read_sensors()
        
        # 자세 업데이트
        self.update_angles(ax, ay, az, gx, gy, gz, mx, my, mz, dt)
        
        # 비중력 가속도 계산
        non_grav_accel = self.calculate_non_gravity_accel(ax, ay, az, self.roll, self.pitch, dt)
        
        # 속도·위치 업데이트
        total_accel, is_stationary = self.update_motion(non_grav_accel, dt)
        
        # 히스토리 저장
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.angle_history.append([self.roll, self.pitch, self.yaw])
        
        # 콘솔 출력
        status = "STATIONARY" if self.stationary_count > STATIONARY_COUNT_LIMIT else f"MOVING({total_accel:.2f})"
        
        print(f"RPY: {self.roll:6.2f}deg {self.pitch:6.2f}deg {self.yaw:6.2f}deg | "
              f"Accel: {non_grav_accel[0]:7.3f} {non_grav_accel[1]:7.3f} {non_grav_accel[2]:7.3f} | "
              f"Vel: {self.velocity[0]:7.3f} {self.velocity[1]:7.3f} {self.velocity[2]:7.3f} | "
              f"Pos: {self.position[0]:7.3f} {self.position[1]:7.3f} {self.position[2]:7.3f} | "
              f"{status}")
        
        return True

    """ IMU 데이터수집 시작 """    
    def start(self):
        
        if not self.init_mpu():
            return False
        
        time.sleep(1)
        
        if not self.calibrate_gyro():
            return False
        if not self.calibrate_accel():
            return False
        if not self.calibrate_mag():
            return False
        
        self.initialize_pose()
        print("\nInitialization complete!")
        print("Format: Roll(??) Pitch(??) Yaw(??) | AccelX AccelY AccelZ(m/s??) | VelX VelY VelZ(m/s) | PosX PosY PosZ(m) | Status")
        
        self.running = True
        return True

    """ IMU 데이터수집 종료 """   
    def stop(self):
        
        self.running = False

class IMUVisualizer:
    """ 3D visualization class for IMU data """
    
    def __init__(self, imu):
        self.imu = imu
        self.fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(221, projection='3d')  # 3D position
        self.ax2 = self.fig.add_subplot(222)                   # Velocity
        self.ax3 = self.fig.add_subplot(223)                   # Angles
        self.ax4 = self.fig.add_subplot(224)                   # Acceleration
        
        self.setup_plots()
        
        # Animation
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=50, blit=False)
    
    def setup_plots(self):
        """Setup all plot configurations"""
        # 3D Position plot
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title('3D Position Tracking')
        
        # Velocity plot
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Velocity (m/s)')
        self.ax2.set_title('Velocity vs Time')
        self.ax2.legend(['X', 'Y', 'Z'])
        self.ax2.grid(True)
        
        # Angles plot
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Angle (degrees)')
        self.ax3.set_title('Orientation (Roll, Pitch, Yaw)')
        self.ax3.legend(['Roll', 'Pitch', 'Yaw'])
        self.ax3.grid(True)
        
        # Acceleration plot
        self.ax4.set_xlabel('Time')
        self.ax4.set_ylabel('Acceleration (m/s??)')
        self.ax4.set_title('Non-Gravity Acceleration')
        self.ax4.legend(['X', 'Y', 'Z'])
        self.ax4.grid(True)
    
    def update_plots(self, frame):
        """Update all plots with new data"""
        if not self.imu.position_history:
            return
        
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Get data
        positions = list(self.imu.position_history)
        velocities = list(self.imu.velocity_history)
        angles = list(self.imu.angle_history)
        
        if len(positions) < 2:
            return
        
        # 3D Position plot
        pos_array = np.array(positions)
        self.ax1.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'b-', alpha=0.7)
        self.ax1.scatter(pos_array[-1, 0], pos_array[-1, 1], pos_array[-1, 2], 
                        c='red', s=50, label='Current')
        self.ax1.scatter(pos_array[0, 0], pos_array[0, 1], pos_array[0, 2], 
                        c='green', s=50, label='Start')
        
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title('3D Position Tracking')
        self.ax1.legend()
        
        # Auto-scale with some padding
        if len(pos_array) > 0:
            max_range = np.array([pos_array[:, 0].max() - pos_array[:, 0].min(),
                                pos_array[:, 1].max() - pos_array[:, 1].min(),
                                pos_array[:, 2].max() - pos_array[:, 2].min()]).max() / 2.0
            mid_x = (pos_array[:, 0].max() + pos_array[:, 0].min()) * 0.5
            mid_y = (pos_array[:, 1].max() + pos_array[:, 1].min()) * 0.5
            mid_z = (pos_array[:, 2].max() + pos_array[:, 2].min()) * 0.5
            
            if max_range > 0:
                self.ax1.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax1.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Velocity plot
        vel_array = np.array(velocities)
        time_axis = range(len(vel_array))
        self.ax2.plot(time_axis, vel_array[:, 0], 'r-', label='X')
        self.ax2.plot(time_axis, vel_array[:, 1], 'g-', label='Y')
        self.ax2.plot(time_axis, vel_array[:, 2], 'b-', label='Z')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Velocity (m/s)')
        self.ax2.set_title('Velocity vs Time')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Angles plot
        angle_array = np.array(angles)
        self.ax3.plot(time_axis, angle_array[:, 0], 'r-', label='Roll')
        self.ax3.plot(time_axis, angle_array[:, 1], 'g-', label='Pitch')
        self.ax3.plot(time_axis, angle_array[:, 2], 'b-', label='Yaw')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Angle (degrees)')
        self.ax3.set_title('Orientation (Roll, Pitch, Yaw)')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Non-gravity acceleration (from current filtered values)
        accel_data = [[self.imu.accel_filtered[0], self.imu.accel_filtered[1], self.imu.accel_filtered[2]] 
                     for _ in range(len(time_axis))]
        accel_array = np.array(accel_data)
        self.ax4.plot(time_axis, accel_array[:, 0], 'r-', label='X')
        self.ax4.plot(time_axis, accel_array[:, 1], 'g-', label='Y')
        self.ax4.plot(time_axis, accel_array[:, 2], 'b-', label='Z')
        self.ax4.set_xlabel('Time')
        self.ax4.set_ylabel('Acceleration (m/s??)')
        self.ax4.set_title('Non-Gravity Acceleration')
        self.ax4.legend()
        self.ax4.grid(True)
        
        plt.tight_layout()

def main():
    """ Main function """
    print("Enhanced IMU Motion Tracking for Raspberry Pi with 3D Visualization")
    print("Make sure your MPU6050/MPU9250 is connected to I2C bus 1")
    print("Press Ctrl+C to exit")
    
    try:
        # Initialize IMU
        imu = IMU(bus_number=1)
        
        if not imu.start():
            print("Failed to initialize IMU")
            return
        
        # Create visualizer
        visualizer = IMUVisualizer(imu)
        
        # Start IMU data collection in a separate thread
        def imu_thread():
            while imu.running:
                imu.update()
                time.sleep(0.001)  # Small delay to prevent overwhelming the system
        
        thread = threading.Thread(target=imu_thread, daemon=True)
        thread.start()
        
        # Show plots
        plt.show()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'imu' in locals():
            imu.stop()

if __name__ == "__main__":
    main()