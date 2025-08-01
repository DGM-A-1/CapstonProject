#!/usr/bin/env python
# -*- coding: utf-8 -*-

##################################################################################
# Delivery Control Node                                                          #
# - Publishes /running while navigating                                          #
# - Uses is_reached flag to wait for arrival instead of wait_for_result()         #
##################################################################################

import rospy
import sys
import actionlib
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion, Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from std_msgs.msg import Int8, Bool

# 상태 변수
x_meter = y_meter = 0.0
start_room = end_room = 0
is_reached = False
running = False
weight_alert = False

# Threshold
THRESHOLD_KG = 5.0

# 목표 포인트
home_point   = (-32.5929, 46.9838, 0.0, 1.0)
room1_point = (  4.3085, 45.4902,-0.7210,0.692831)
room2_point = (  2.3756,  7.3587,-0.7329,0.68028 )
room3_point = (-36.1953,  6.3922, 0.7144,0.69970 )
point_array = [home_point, room1_point, room2_point, room3_point]

# Publishers
running_pub = None

# Subscribers' callbacks
def getAmclPose(msg):
    global x_meter, y_meter
    x_meter = msg.pose.pose.position.x
    y_meter = msg.pose.pose.position.y

def getStartRoom(msg):
    global start_room
    start_room = msg.data

def getEndRoom(msg):
    global end_room
    end_room = msg.data

def getNavStatus(msg):
    global is_reached, running
    if msg.status.status == 3:  # SUCCEEDED
        is_reached = True
        running = False
        running_pub.publish(Bool(data=False))
        rospy.loginfo("[Delivery] 이동 완료, running=False")

def weightAlertCallback(msg):
    global weight_alert
    if running:
        return
    weight_alert = msg.data
    rospy.loginfo("[weight_alert] loaded → %s", weight_alert)

# 네비게이션 goal 퍼블리시 (비동기)
def pubNavGoal(pose):
    global running, is_reached
    running = True
    is_reached = False
    running_pub.publish(Bool(data=True))
    rospy.loginfo("[Delivery] running=True: 이동 시작")

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    client.wait_for_server()
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    # pose 튜플 unpack
    x, y, z, w = pose if len(pose)==4 else (*pose, 0.0, 1.0)
    goal.target_pose.pose = Pose(Point(x, y, 0), Quaternion(0,0,z,w))
    client.send_goal(goal)

# 목적지까지 대기

def move_to(idx, rate):
    pubNavGoal(point_array[idx])
    rospy.loginfo("[Delivery] point%d으로 이동 중…", idx)
    while not is_reached and not rospy.is_shutdown():
        rate.sleep()
    rospy.loginfo("[Delivery] point%d 도착 완료", idx)

# 무게 대기

def waitWeightLoaded():
    rospy.loginfo("▶ 대기: 무게 ≥%.2fkg 감지 중…", THRESHOLD_KG)
    while not weight_alert and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("✔ 무게 ≥%.2fkg 감지!", THRESHOLD_KG)

def waitWeightUnloaded():
    rospy.loginfo("▶ 대기: 무게 < %.2fkg 감지 중…", THRESHOLD_KG)
    while weight_alert and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("✔ 무게 < %.2fkg 감지!", THRESHOLD_KG)

# 메인
if __name__ == '__main__':
    rospy.init_node("delivery_ctrl_node")
    rate = rospy.Rate(10)

    # 퍼블리셔 초기화
    running_pub = rospy.Publisher('/running', Bool, queue_size=1)

    # 구독자 초기화
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, getAmclPose)
    rospy.Subscriber('/start_room', Int8, getStartRoom)
    rospy.Subscriber('/end_room', Int8, getEndRoom)
    rospy.Subscriber('/move_base/result', MoveBaseActionResult, getNavStatus)
    rospy.Subscriber('/weight_alert', Bool, weightAlertCallback)

    try:
        rospy.loginfo("[Delivery] HOME에서 /start_delivery 대기 중…")
        rospy.wait_for_message('/start_delivery', Bool)
        rospy.loginfo("✔ /start_delivery=True 수신! 시나리오 시작")

        # point1 (상차)
        move_to(1, rate)
        # 무게 감지
        waitWeightLoaded()

        # room2, room3 순차 이동
        for idx in [2, 3]:
            move_to(idx, rate)

        # 하차 전 대기
        rospy.loginfo("[Delivery] Point3 도착 후 5초 대기…")
        rospy.sleep(5.0)
        waitWeightUnloaded()

        # HOME 귀환
        move_to(0, rate)
        rospy.loginfo("[Delivery] HOME 복귀 완료. 시나리오 종료.")

        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("✋ ROSInterruptException: 종료 신호 수신")
    except KeyboardInterrupt:
        rospy.loginfo("✋ KeyboardInterrupt: Ctrl+C 입력, 안전 종료")
    finally:
        rospy.loginfo("▶ delivery_ctrl_node 종료")
        rospy.signal_shutdown("shutting down")
