import cv2
import mediapipe as mp
import numpy as np
import threading
from modules.angle_calculation import calculate_angle
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 俯卧撑
def detect_pushup():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose_estimator = PoseEstimator(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    pushup_count = 0
    stage = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            left_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # 显示角度
            cv2.putText(image, f"Elbow Angle: {elbow_angle:.1f}",
                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 判断推起和下压阶段
            if elbow_angle > 160:
                stage = "up"
            if elbow_angle < 90 and stage == "up":
                stage = "down"
                pushup_count += 1
                threading.Thread(target=speak, args=(f"完成第{pushup_count}次俯卧撑",)).start()

            # 显示计数
            cv2.putText(image, f"Count: {pushup_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        pose_estimator.draw_landmarks(image)
        cv2.imshow("俯卧撑检测", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
