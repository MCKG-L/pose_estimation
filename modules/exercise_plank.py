import cv2
import mediapipe as mp
import numpy as np
import time
from modules.angle_calculation import calculate_angle, calculate_torso_angle
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 平板支撑
def detect_plank():
    cap = cv2.VideoCapture(0)
    pose_estimator = PoseEstimator()
    mp_pose = mp.solutions.pose

    start_time = None
    timer_running = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 获取关键点坐标
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # 计算躯干角度和髋关节角度
                torso_angle = calculate_torso_angle(shoulder, hip)
                hip_angle = calculate_angle(shoulder, hip, ankle)

                # 姿势判断
                if 160 < hip_angle < 190 and torso_angle < 10:
                    if not timer_running:
                        start_time = time.time()
                        timer_running = True
                    elapsed = int(time.time() - start_time)
                    cv2.putText(image, f"Time: {elapsed}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    timer_running = False
                    cv2.putText(image, "Adjust Posture!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")

        pose_estimator.draw_landmarks(image)
        cv2.imshow('Plank Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()