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
from scipy.spatial.transform import Rotation

import logging
import threading

GRAVITY = 9.80665
# 이 칼만필터는 Yaw(heading) 추정에만 사용됨.
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
        self.accel_coeffs = [np.array([0.99814955, -0.13019077]), # 가속도계 보정 계수 (mx, bx)
                             np.array([0.99804448, 0.2620614]), # 가속도계 보정 계수 (my, by)
                             np.array([0.98654515, 0.1548281])] # 가속도계 보정 계수 (mz, bz)
    
        self.gyro_coeffs  = np.array([0.3926849365234375, -2.9891204833984375, 0.9511947631835938])
        self.mag_coeffs = np.array([ 48.90632629394531,  0.318145751953125, 36.475467681884766]) #[uT]
        
        # SG 필터 윈도우 초기화
        self._accel_window = deque(maxlen=51)
        
        # 이전 가속도, 속도 값 초기화
        # 증분(trapezoidal) 적분 시, 이전 가속도·속도 값이 필요
        # 초기 : Euler 적분: v = v_prev + a_curr * dt => a_curr에 노이즈가 낄 경우 값이 튀는 문제
        # 현재 : trapz 적분: v = v_prev + (a_prev + a_curr) * dt / 2 => a_prev와 a_curr의 평균을 사용
        self._prev_accel = np.zeros(3)
        self._prev_vel   = np.zeros(3)

        self.velocity = np.zeros(3)  # 초기 속도
        self.position = np.zeros(3)  # 초기 위치
        self.accel_filtered = np.zeros(3) # 초기 가속도

        # Roll Pitch => 가속도계랑 퓨전하면 오히려 정확도 떨어져버릴지도
        #self.kf_roll = Kalman()
        #self.kf_pitch = Kalman()
        self.kf_yaw = Kalman()
        
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # LPF 파라미터: 컷오프 주파수(fc) 설정
        self.fs   = 100.0           # 샘플링 주파수(Hz)
        self.fc   = 20.0             # 컷오프 주파수(Hz), 필요에 따라 조절
        self.dt   = 1.0 / self.fs

        # RC = 1/(2πfc), alpha = dt/(RC+dt)
        self.RC      = 1.0 / (2 * math.pi * self.fc)
        self.alpha_l = self.dt / (self.RC + self.dt)

        # LPF 상태 초기값
        self.accel_lpf = np.zeros(3)

        self.count = 0  # 샘플링 횟수 카운트
    
        self.vel_body = np.zeros(3)  # 바디 프레임 속도
        self.pos_body = np.zeros(3)  # 바디 프레임 위치
        
        self.running = False
        self.thread = None
        self.accel = np.zeros(3)
        
        self._rot_enu_to_ned = Rotation.from_dcm(
            np.array([
                [0, 1, 0],  # E->N
                [1, 0, 0],  # N->E
                [0, 0, -1]   # U->D
            ]))
        
        # self.yaw_offset = 0.0
        self._accel_zeroed_window = deque(maxlen=51) # 드리프트 보정된 가속도 저장
    
    '''
    센서 값 읽기 및 보정 함수
    - 기능
     1) MPU9250 센서 값 읽기
     2) 가속도계 보정: calibrated = m * raw + b
     3) 자이로 보정: calibrated = raw - offset
     4) 자력계 보정: calibrated = raw - offset
    
    주의사항 => 이 함수말고는 전부 이 함수를 거쳐서 센서 값을 읽어야함
    '''
    # 센서 값 읽기 및 보정 함수
    def read_sensors(self):
        # 1) 원시값 읽기[g, degree/sec, uT]
        ax, ay, az, gx, gy, gz = mpu6050_conv()
        mx, my, mz             = AK8963_conv()

        # 2) 가속도 보정: calibrated = m * raw + b
        a_raw = np.array([ax, ay, az])
        a_cal = np.zeros(3)
        for i, (m, b) in enumerate(self.accel_coeffs):
            a_cal[i] = m * a_raw[i] + b

        # 3) 자이로 보정: calibrated = raw - bias
        g_raw = np.array([gx, gy, gz])
        g_cal = g_raw - self.gyro_coeffs
        
        # 4) 자력계 보정: calibrated = raw - offset
        m_raw = np.array([mx, my, mz])
        m_cal = m_raw - self.mag_coeffs

        # 5) 반환값 = 가속도, 자이로, 자력계 보정이 완료된 센서 값
        return a_cal, g_cal, m_cal
    
    '''
    자세 초기화 함수
    Setup 단계에서 한 번만 호출
    - 기능
     1) 센서 값 읽어오기
     2) 가속도계로 Roll, Pitch 계산
     3) 자력계로 Yaw 계산
    '''

    # 초기 자세 함수 -> 한번만 호출, Roll, Pitch는 가속도계로 Yaw로 지자계로 계산
    def initialize_pose(self):
        # 1) 센서 값 읽어 오기
        acc, gyro, mag = self.read_sensors()
        ax, ay, az = acc[0], acc[1], acc[2] #[g]
        gx, gy, gz = gyro[0], gyro[1], gyro[2] #[degree/sec]
        mx, my, mz = mag[0], mag[1], mag[2] #[uT]
        
        # 2) 가속도계로 Roll Pitch 계산 [degrees] -> 원래는 rad단위인데 deg단위로 통일
        acc_roll = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
        self.roll = acc_roll
        self.pitch = acc_pitch
        
        # 3) 지자계로 Yaw 계산 [degrees]
        self.yaw = self.compute_tilt_heading(mx, my, mz, acc_roll, acc_pitch)
        self.kf_yaw.angle = self.yaw

    ''' 
    지자계로 Yaw 계산
    - 기능
        1) Roll, Pitch를 토대로 Yaw 계산
    '''

    def compute_tilt_heading(self, mx, my, mz, roll_deg, pitch_deg):
        roll_rad = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)
        
        xh = mx * math.cos(pitch_rad) + mz * math.sin(pitch_rad)
        yh = (mx * math.sin(roll_rad) * math.sin(pitch_rad) + 
              my * math.cos(roll_rad) - 
              mz * math.sin(roll_rad) * math.cos(pitch_rad))
        
        # 원래는 rad단위인데 deg단위로 통일
        heading = math.degrees(math.atan2(yh, xh)) #[degrees]
        
        # 0-360도 범위로 정규화
        if heading < 0:
            heading += 360
        elif heading >= 360:
            heading -= 360
        return heading

    """
    내부용: Roll(R), Pitch(P), Yaw(Y)(deg)로부터 회전행렬 T 를 계산
    """
    def _get_euler_rate_transformation_matrix(self, R_deg, P_deg, Y_deg, seq='zyx'):
        # 각도 → 라디안
        R = math.radians(R_deg)
        P = math.radians(P_deg)
        Y = math.radians(Y_deg)

        # 사인, 코사인, 탄젠트
        sR, cR = math.sin(R), math.cos(R)
        sP, cP = math.sin(P), math.cos(P)
        sY, cY = math.sin(Y), math.cos(Y)

        # 작은 cos 제거 (gimbal lock 회피)
        if abs(cP) < 1e-2:
            cP = 1e-2 if cP >= 0 else -1e-2

        seq = seq.lower()
        if seq == 'zyx':
            # Z–Y–X extrinsic (Yaw–Pitch–Roll)
            T = np.array([
                [1,       sR * math.tan(P),  cR * math.tan(P)],
                [0,               cR,              -sR      ],
                [0,      sR / cP,           cR / cP       ]
            ])
        else:
            raise ValueError("지원하지 않는 시퀀스: " + seq)

        # 반환: 변환 행렬 T
        return T
    
    """
    바디 프레임 기준 각속도 (p, q, r) [degree/s]를 받아
    월드 프레임 Euler 각의 시간 미분 (φ̇, θ̇, ψ̇) [degree/s]를 반환
    """
    # 회전 행렬 계산 함수 (SciPy 라이브러리 사용)
    def get_rotation_matrix(self, roll, pitch, yaw):
        """
        SciPy를 이용해 Extrinsic Z-Y-X 순서 (Yaw-Pitch-Roll) 회전 행렬 생성
        :param roll: Roll 각도 (deg)
        :param pitch: Pitch 각도 (deg)
        :param yaw: Yaw 각도 (deg)
        :return: 3x3 회전 행렬 (world <- body)
        """
        # Extrinsic Z->Y->X 회전: yaw, pitch, roll
        return Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_dcm()
    
    
    def body_rates_to_euler_rate(self, p, q, r, roll_deg, pitch_deg, yaw_deg, seq='zyx'):
        # 1) 변환 행렬 T 구하기
        T = self._get_euler_rate_transformation_matrix(roll_deg, pitch_deg, yaw_deg, seq)

        # 2) T 행렬 곱해서 Euler rate 구하기
        omega_body = np.array([p, q, r], dtype=float)
        euler_rates = T.dot(omega_body)

        # 반환: 오일러 각속도 반환 (roll_rate, pitch_rate, yaw_rate) [degree/s]
        return tuple(euler_rates)

    """
    최근 self._accel_window에 쌓인 비중력 가속도 변화량(range)이
    threshold(m/s²) 이하이면, velocity를 0으로 리셋.
    """
    def zero_velocity_update(self, threshold=0.05):
        if len(self._accel_window) < self._accel_window.maxlen:
            return  # 윈도우가 아직 채워지지 않음
        arr = np.vstack(self._accel_window)       # shape = (window, 3)
        peak2peak = np.ptp(arr, axis=0)           # 각 축별 max-min
        if np.all(peak2peak < threshold * GRAVITY):
            # 정지 상태로 판단 → 속도 리셋
            self.velocity[:]   = 0.0
            self._prev_vel[:]  = 0.0
    
    """별도 스레드에서 실행될 센서 업데이트 루프"""
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        return False
    
    """센서 읽기 스레드를 중지합니다."""
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def get_yaw(self):
        return self.yaw

    '''
    IMU 센서 기능 통합 함수
    main loop에서 이 함수를 호출
    - 기능
     1) 가속도 -> 속도 -> 변위 증분적분으로 계산
     2) Roll, Pitch, Yaw 업데이트

    - 주의사항
     1) 이때 가속도는 Savitzky-Golay 필터링을 거쳐야 함
     2) Yaw는 Kalman 필터를 통해 업데이트
     3) Roll, Pitch는 오일러 각속도 값을 적분해 업데이트
     => 가속도계가 외부 진동에 취약해 칼만필터로 업데이트 시 오히려 정확도가 떨어짐
    '''

    ''' ★★★★★★★★★★★★★★★★★★ 중요 ★★★★★★★★★★★★★★★★★★★★ '''
    def _update(self):
        SAMPLE_RATE = 100.0  #100 Hz
        DT = 1.0 / SAMPLE_RATE # 0.01 초 주기로 갱신
        
        # 0) warm-up 추가 : 0.5초 동안 센서 버퍼를 미리 채워서 SG 필터가 즉시 적용되도록 만듦
        for _ in range(self._accel_window.maxlen):
            a_cal, g_cal, m_cal = self.read_sensors()
            rot_mat =  self.get_rotation_matrix(self.roll, self.pitch, self.yaw)
            a_world = rot_mat @ (a_cal * GRAVITY)
            a_non_gravity = a_world - np.array([0,0,GRAVITY])
            self._accel_window.append(a_non_gravity)
            time.sleep(DT)
        
        # time.perf_counter()는 고해상도 타이머로, sleep()과 함께 사용하여 정확한 타이밍 유지
        next_time = time.perf_counter()

        print("Starting real-time 100Hz update loop…")
        while self.running:
            # 스케줄링
            next_time += DT

            # 1) 보정된 센서 읽기
            a_cal, g_cal, m_cal = self.read_sensors()

            # 2) 각속도 body frame -> world frame
            # g_cal = [p, q, r] (deg/s)
            # *g_cal = g_cal[0], g_cal[1], g_cal[2]
            g_world = self.body_rates_to_euler_rate(*g_cal, self.roll, self.pitch, self.yaw)
            
            self.roll  += g_world[0] * DT
            self.pitch += g_world[1] * DT

            # 3) Yaw: 칼만 필터
            # Measurement: 자력계로부터 Yaw 계산
            # Expectation: 각속도계 -> 오일러 각속도
            meas_yaw = self.compute_tilt_heading(*m_cal, self.roll, self.pitch)
            self.yaw = self.kf_yaw.update(meas_yaw, g_world[2], DT)
            
            ############### 자세 갱신 완료 ###############
            
        
            ''' 이 밑 부분부터는 필요가 없어졌음 '''
            ############## 가속도 적분 시작 ##############            
            # # 4) a_filt(body frame) -> a_world(world frame) -> a_world_non_gravity
            # # 좌표계 변환 : a_world = R @ a_filt
            # # 중력 제거: a_non_gravity = a_world - g
            # rot_mat = self.get_rotation_matrix(self.roll, self.pitch, self.yaw) # [ENU]
            # g = np.array([0, 0, GRAVITY]) #[m/s²]
            # a_world = rot_mat @ (np.array(a_cal).T * GRAVITY) #[m/s²]
            # a_non_gravity_enu = a_world - g #[m/s²]

            # # 추가) ENU -> NED 변환
            # a_non_gravity = self._rot_enu_to_ned.apply(a_non_gravity_enu)  # ENU -> NED 변환
            
            # 4) 단순 전진 가속도만 사용
            a_forward = max(0.0,a_cal[0] *GRAVITY)

            yaw_rad = math.radians(self.yaw)
            ax = a_forward * math.cos(yaw_rad)
            ay = a_forward * math.sin(yaw_rad)

            a_non_gravity = np.appay([ax,ay,0.0])

            # 추가) accel_zeord 계산
            self._accel_window.append(a_non_gravity) 
            drift = np.mean(self._accel_window, axis=0)
            accel_zeroed = a_non_gravity - drift
            
            self._accel_zeroed_window.append(accel_zeroed)
            
            # 4) 가속도 필터 적용
            if len(self._accel_zeroed_window) == self._accel_zeroed_window.maxlen:
                arr = np.vstack(self._accel_zeroed_window)
                arr_sg = savgol_filter(arr, 51, 3, axis=0, mode='mirror')
                
                #coeffs = savgol_coeffs(51, 3, deriv=0, delta=1.0, pos=50, use='conv')
                #a_filt = np.dot(coeffs, np.vstack(self._accel_window))
                a_filt = arr_sg[-1]
            else:
                a_filt = a_non_gravity
            
            #추가) 가속도 LPF 적용
            self.accel_lpf = ( self.alpha_l * a_filt
                         + (1 - self.alpha_l) * self.accel_lpf)
            
            
            # 추가) ZUPT 검사 알고리즘 추가
            self.count += 1
            if self.count >= 200:
                
                self.zero_velocity_update(threshold=0.05)
                self.count = 0

            # 5) 속도, 변위, 가속도 state 업데이트
            # 속도 변위 => 증분 트라페조이드 적분
            self.accel = self.accel_lpf.copy()
            self.velocity = self._prev_vel + (self._prev_accel + self.accel) * (DT/2)
            self.position = self.position     + (self._prev_vel   + self.velocity) * (DT/2)
            
            # 6) 이전 값 업데이트
            self._prev_accel = self.accel.copy()
            self._prev_vel   = self.velocity.copy()
            
            # 7) 바디 프레임 속도, 위치 계산
            rot_mat = self.get_rotation_matrix(self.roll, self.pitch, self.yaw)
            R_w2b = rot_mat.T
            vel_body = R_w2b.dot(self.velocity)    # Body-frame 속도
            pos_body = R_w2b.dot(self.position)    # Body-frame 위치
            
            self.vel_body = vel_body
            self.pos_body = pos_body
            
            ############## 가속도 적분 완료 ##############
            # 7) 루프 타이밍 보정
            now = time.perf_counter()
            sleep_t = next_time - now
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # 스케줄 지연 발생 시 로깅 또는 대체 처리
                pass
            

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    imu = IMU()
    imu.initialize_pose()
    
    # ——— 디버깅용 데이터 버퍼 초기화 ———
    max_samples = 1000  # 예: 10초치(100Hz * 10s)
    accel_raw_buf     = np.zeros((max_samples, 3))
    accel_filt_buf    = np.zeros((max_samples, 3))
    velocity_buf      = np.zeros((max_samples, 3))
    position_buf      = np.zeros((max_samples, 3))
    orientation_buf   = np.zeros((max_samples, 3))  # roll, pitch, yaw (KF)
    meas_yaw_buf      = np.zeros(max_samples)      # magnetometer-only yaw
    int_yaw_buf       = np.zeros(max_samples)      # gyro-integrated yaw
    timestamps        = np.zeros(max_samples)
    
    SAMPLE_RATE = 100.0
    DT = 1.0 / SAMPLE_RATE
    
    sample_count = 0
    samples_per_zvu = int(2 * SAMPLE_RATE)  # 200
    
    # ——— 0.5초 워밍업: 버퍼 미리 채우기 ———
    for _ in range(imu._accel_window.maxlen):
        a_cal, g_cal, m_cal = imu.read_sensors()
        
        # world-frame + 중력 제거
        rot_mat = imu.get_rotation_matrix(imu.roll, imu.pitch, imu.yaw)
        a_world = rot_mat @ (a_cal * GRAVITY)
        a_non_gravity = a_world - np.array([0,0,GRAVITY])
        
        imu._accel_window.append(a_non_gravity)
        time.sleep(DT)     # 실제 센서 주기 맞춰서
    
    next_time = time.perf_counter()
    yaw_int_prev = imu.yaw
    
    print("▶️ 버퍼 워밍업 완료. 이제부터 SG 필터가 즉시 적용됩니다.")
    print("Starting real-time 100Hz update loop with logging…")
    for i in range(max_samples):
        next_time += DT
        a_cal, g_cal, m_cal = imu.read_sensors()
        rot_mat = imu.get_rotation_matrix(imu.roll, imu.pitch, imu.yaw)
        
        g = np.array([0, 0, GRAVITY])    
        a_world = rot_mat @ (np.array(a_cal).T * GRAVITY)
        
        a_non_gravity_ENU = a_world - g
        # 추가) ENU -> NED 변환
        a_non_gravity = imu._rot_enu_to_ned.apply(a_non_gravity_ENU)


                # —— 제로링 단계 —— 
        imu._accel_window.append(a_non_gravity)
        drift = np.mean(np.vstack(imu._accel_window), axis=0)
        accel_zeroed = a_non_gravity - drift
        imu._accel_zeroed_window.append(accel_zeroed)

        if len(imu._accel_zeroed_window) == imu._accel_zeroed_window.maxlen:
            arr = np.vstack(imu._accel_zeroed_window)
            #coeffs = savgol_coeffs(51, 3, deriv=0, delta=1.0, pos=50, use='conv')
            #a_filt = np.dot(coeffs, np.vstack(imu._accel_window))
            arr_sg = savgol_filter(arr, 51, 3, axis=0, mode='mirror')
            a_filt = arr_sg[-1]
        else:
            a_filt = a_non_gravity
        
        imu.accel_lpf = ( imu.alpha_l * a_filt
                         + (1 - imu.alpha_l) * imu.accel_lpf )
        

        sample_count += 1
        if sample_count >= samples_per_zvu:
            imu.zero_velocity_update(threshold=0.05)
            sample_count = 0
        
        vel = imu._prev_vel + (imu._prev_accel + imu.accel_lpf) * (DT/2)
        pos = imu.position + (imu._prev_vel + vel) * (DT/2)

        # 자세 업데이트 (roll, pitch)
        g_world = imu.body_rates_to_euler_rate(*g_cal, imu.roll, imu.pitch, imu.yaw)
        imu.roll  += g_world[0] * DT
        imu.pitch += g_world[1] * DT
        
        # Raw yaw from magnetometer
        meas_yaw = imu.compute_tilt_heading(*m_cal, imu.roll, imu.pitch)
        
        # Integrated yaw from gyro
        yaw_int = yaw_int_prev + g_world[2] * DT
        yaw_int_prev = yaw_int
        
        # Kalman-fused yaw
        imu.yaw = imu.kf_yaw.update(meas_yaw, g_world[2], DT)

        # Update buffers
        accel_raw_buf[i]   = a_non_gravity
        accel_filt_buf[i]  = imu.accel_lpf
        velocity_buf[i]    = vel
        position_buf[i]    = pos
        orientation_buf[i] = np.array([imu.roll, imu.pitch, imu.yaw])
        meas_yaw_buf[i]    = meas_yaw
        int_yaw_buf[i]     = yaw_int
        timestamps[i]      = i * DT
        
        # ——— 상태 업데이트 ———
        imu._prev_accel = imu.accel_lpf.copy()
        imu._prev_vel   = vel.copy()
        imu.position    = pos.copy()
        

        # 루프 타이밍 유지
        now = time.perf_counter()
        sleep_t = next_time - now
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ——— 로그된 데이터로 플롯 생성 ———
    print(len(timestamps) / 10, "sampling rate")
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # 1) 가속도 (필터 전·후)
    for idx, label in zip([0,1,2], ['X','Y','Z']):
        axes[0].plot(timestamps, accel_filt_buf[:,idx],      label=f'Accel_{label} (filtered)')
        axes[0].plot(timestamps, accel_raw_buf[:,idx], alpha=0.3, linestyle='--',
                    label=f'Accel_{label} (raw)')
    axes[0].set_ylabel('Accel (m/s²)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # 2) 속도
    for idx, label in zip([0,1,2], ['X','Y','Z']):
        axes[1].plot(timestamps, velocity_buf[:,idx], label=f'Vel_{label}')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    # 3) 변위
    for idx, label in zip([0,1,2], ['X','Y','Z']):
        axes[2].plot(timestamps, position_buf[:,idx], label=f'Pos_{label}')
    axes[2].set_ylabel('Displacement (m)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    # 4) 자세 (roll, pitch, yaw)
    axes[3].plot(timestamps, orientation_buf[:,0], label='Roll')
    axes[3].plot(timestamps, orientation_buf[:,1], label='Pitch')
    axes[3].plot(timestamps, orientation_buf[:,2], label='Yaw (KF)')
    # faint raw and integrated yaw
    axes[3].plot(timestamps, meas_yaw_buf, alpha=0.3, linestyle='--', label='Yaw_raw (mag)')
    axes[3].plot(timestamps, int_yaw_buf, alpha=0.3, linestyle=':', label='Yaw_int (gyro)')
    axes[3].set_ylabel('Angle (deg)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()