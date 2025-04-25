import cv2
import mediapipe as mp
import numpy as np
import threading
from modules.angle_calculation import calculate_angle, calculate_torso_angle
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 仰卧起坐
def detect_situp():
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose_estimator = PoseEstimator(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    situp_count = 0
    is_lying_down = False
    max_torso_angle = 0
    initial_hip_height = None

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
            left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            if initial_hip_height is None:
                initial_hip_height = (left_hip[1] + right_hip[1]) / 2

            torso_angle = calculate_torso_angle(left_shoulder, left_hip)

            if torso_angle > 35:
                if not is_lying_down:
                    max_torso_angle = max(max_torso_angle, torso_angle)
            else:
                if is_lying_down and max_torso_angle > 60:
                    situp_count += 1
                    threading.Thread(target=speak, args=(f"Sit-up {situp_count} completed",)).start()
                max_torso_angle = 0
                is_lying_down = True
            is_lying_down = torso_angle > 25

            cv2.putText(image, f"Torso Angle: {torso_angle:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Count: {situp_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        pose_estimator.draw_landmarks(image)
        cv2.imshow('Sit-up Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
