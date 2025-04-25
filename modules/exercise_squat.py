import cv2
import mediapipe as mp
import numpy as np
import threading
from modules.angle_calculation import calculate_angle
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 深蹲检测
def detect_squat():
    cap = cv2.VideoCapture(0)
    pose_estimator = PoseEstimator()
    mp_pose = mp.solutions.pose

    squat_count = 0
    stage = None

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
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # 计算膝关节角度
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # 显示角度
                cv2.putText(image, f"Knee Angle: {knee_angle:.1f}",
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 深蹲逻辑判断
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    squat_count += 1
                    threading.Thread(target=speak, args=(f"深蹲 {squat_count} 次",)).start()

                # 显示计数
                cv2.putText(image, f"Count: {squat_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error: {e}")

        pose_estimator.draw_landmarks(image)
        cv2.imshow('Squat Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()