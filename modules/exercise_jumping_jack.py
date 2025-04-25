import time
import cv2
import mediapipe as mp
import numpy as np
import threading
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 开合跳检测
def detect_jumping_jack():
    cap = cv2.VideoCapture(0)
    pose_estimator = PoseEstimator()
    mp_pose = mp.solutions.pose

    jumping_jack_count = 0
    initial_feet_distance_threshold = None
    is_open = False
    consecutive_closed_frames = 0
    closed_threshold = 5  # 需要连续多少帧腿部靠近才认为闭合

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

                # 获取左右脚踝或膝盖的水平坐标
                left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
                right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
                feet_distance = abs(right_ankle_x - left_ankle_x)

                # 初始化初始距离阈值（可选，用于更鲁棒的检测）
                if initial_feet_distance_threshold is None:
                    initial_feet_distance_threshold = feet_distance * 0.3  # 可以调整这个比例

                # 检测是否张开
                if feet_distance > (initial_feet_distance_threshold if initial_feet_distance_threshold else 0.1): # 调整第二个阈值
                    if not is_open:
                        is_open = True
                        consecutive_closed_frames = 0 # Reset closed frame counter
                else:
                    consecutive_closed_frames += 1
                    if is_open and consecutive_closed_frames > closed_threshold:
                        jumping_jack_count += 1
                        threading.Thread(target=speak, args=(f"开合跳 {jumping_jack_count} 次",)).start()
                        is_open = False
                        time.sleep(0.5) # 防重复计数

                cv2.putText(image, f"Jumping Jacks: {jumping_jack_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error: {e}")

        pose_estimator.draw_landmarks(image)
        cv2.imshow('Jumping Jack Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_jumping_jack()