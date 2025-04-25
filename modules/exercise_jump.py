import time

import cv2
import mediapipe as mp
import numpy as np
import threading
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 跳跃检测
def detect_jump():
    cap = cv2.VideoCapture(0)
    pose_estimator = PoseEstimator()
    mp_pose = mp.solutions.pose

    jump_count = 0
    prev_ankle_y = 0

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

                # 获取踝关节坐标
                ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y

                # 跳跃检测逻辑
                if abs(ankle_y - prev_ankle_y) > 0.1:  # 阈值可调整
                    jump_count += 1
                    threading.Thread(target=speak, args=(f"跳跃 {jump_count} 次",)).start()
                    time.sleep(0.5)  # 防重复计数

                prev_ankle_y = ankle_y
                cv2.putText(image, f"Jumps: {jump_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error: {e}")

        pose_estimator.draw_landmarks(image)
        cv2.imshow('Jump Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()