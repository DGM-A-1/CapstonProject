#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import rospy, time
import Jetson.GPIO as GPIO
from std_msgs.msg import Bool
from hx711 import HX711
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from collections import deque

# —— 설정 파라미터 ——
DATA_PIN           = rospy.get_param('~data_pin', 7)
CLK_PIN            = rospy.get_param('~clk_pin', 13)
CALIBRATION_FACTOR = rospy.get_param('~calibration_factor', -12000.0)
NUM_SAMPLES        = rospy.get_param('~num_samples', 20)
THRESHOLD_KG       = rospy.get_param('~threshold_kg', 5.0)
HOLD_DURATION      = rospy.get_param('~hold_duration', 3.0)
ALPHA              = rospy.get_param('~alpha', 0.1)
MA_WINDOW          = rospy.get_param('~ma_window', 5)

# —— 전역 상태 변수 ——  
hx                 = None
offset             = 0.0           # 0kg 기준 오프셋
measurement_enabled= False         # 측정 허용 플래그
detection          = False         # 현재 적재 상태
load_start_time    = None
unload_start_time  = None
filtered_weight    = 0.0
ma_buffer          = None
running            = False         # 외부에서 구독할 “실행 중” 플래그

def cleanup():
    GPIO.cleanup()

def on_new_goal(msg):
    """새 /move_base/goal 수신 시: 측정 중지"""
    global measurement_enabled, detection, load_start_time, unload_start_time, ma_buffer, filtered_weight
    measurement_enabled = False
    detection           = False
    load_start_time     = None
    unload_start_time   = None
    ma_buffer.clear()
    filtered_weight     = 0.0
    rospy.loginfo("[HX711] 새 goal 수신 → 측정 중지")

def on_move_result(msg):
    """목표 도착(status=3) 시: 측정 재개"""
    global measurement_enabled, detection, load_start_time, unload_start_time, ma_buffer, filtered_weight
    if msg.status.status == 3:
        measurement_enabled = True
        detection           = False
        load_start_time     = None
        unload_start_time   = None
        ma_buffer.clear()
        filtered_weight     = 0.0
        rospy.loginfo("[HX711] 목표 도착 → 측정 재개")

def on_running(msg):
    """외부 /running 토픽 구독 → True 이면 측정 & 발행 중지"""
    global running
    running = msg.data
    rospy.loginfo("[HX711] /running = %s", running)

def main():
    global hx, offset, measurement_enabled, detection
    global load_start_time, unload_start_time, filtered_weight, ma_buffer

    rospy.init_node('hx711_weight_node')
    rospy.on_shutdown(cleanup)

    pub = rospy.Publisher('/weight_alert', Bool, queue_size=1)
    rate = rospy.Rate(1)  # 1 Hz

    # LPF용 이동평균 버퍼
    ma_buffer = deque(maxlen=MA_WINDOW)

    # GPIO & HX711 초기화
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setmode = lambda *a, **k: None  # hx711 내부 재호출 무력화
    hx = HX711(dout=DATA_PIN, pd_sck=CLK_PIN)
    hx.set_gain(128)

    # — 최초 1회만 오프셋(tare) 측정 —
    rospy.loginfo("== HX711: 초기 Tare offset 측정 ==")
    hx.reset()
    offset = hx.read_average(NUM_SAMPLES)
    rospy.loginfo("  → 초기 오프셋(raw) = %.1f (이 값이 0kg 기준)", offset)

    # 토픽 구독
    rospy.Subscriber('/move_base/goal',   MoveBaseActionGoal,   on_new_goal,    queue_size=1)
    rospy.Subscriber('/move_base/result', MoveBaseActionResult, on_move_result, queue_size=1)
    rospy.Subscriber('/running',          Bool,                 on_running,     queue_size=1)

    rospy.loginfo("== HX711: 준비 완료, /move_base/result(status=3) 대기 중 ==")

    while not rospy.is_shutdown():
        # 1) 아직 도착 전이거나 running 중이면 측정 스킵
        if not measurement_enabled or running:
            rate.sleep()
            continue

        # 2) RAW 무게 읽기 & 음수 방지
        raw_weight = max(0.0, (hx.read_average(NUM_SAMPLES) - offset) / CALIBRATION_FACTOR)

        # 3) 지수 평활 LPF 적용
        filtered_weight = ALPHA * raw_weight + (1 - ALPHA) * filtered_weight

        # 4) 이동평균 적용
        ma_buffer.append(filtered_weight)
        avg_weight = sum(ma_buffer) / len(ma_buffer)

        # 5) 상태 로그
        rospy.loginfo("Weight → raw:%.2f  expLPF:%.2f  MA(%d):%.2f",
                      raw_weight, filtered_weight, len(ma_buffer), avg_weight)

        now = rospy.get_time()
        # 6) 적재 감지
        if avg_weight >= THRESHOLD_KG:
            if load_start_time is None:
                load_start_time = now
            elif not detection and (now - load_start_time) >= HOLD_DURATION:
                detection = True
                pub.publish(Bool(data=True))
                rospy.loginfo("[HX711] 무게 ≥%.2fkg → publish(True)", THRESHOLD_KG)

        # 7) 하차 감지
        else:
            load_start_time = None
            if unload_start_time is None:
                unload_start_time = now
            elif detection and (now - unload_start_time) >= HOLD_DURATION:
                detection = False
                pub.publish(Bool(data=False))
                rospy.loginfo("[HX711] 무게 <%.2fkg → publish(False)", THRESHOLD_KG)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
