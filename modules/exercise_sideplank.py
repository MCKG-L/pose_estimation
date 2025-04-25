import cv2
import mediapipe as mp
import numpy as np
from modules.angle_calculation import calculate_torso_angle
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 侧平板支撑
def detect_sideplank():
    cap = cv2.VideoCapture(0)
    pose_estimator = PoseEstimator()
    mp_pose = mp.solutions.pose

    timer_start = None
    GOOD_POSTURE = False

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

                # 获取双侧肩髋坐标
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # 计算躯干倾斜角度
                shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
                hip_center = np.mean([left_hip, right_hip], axis=0)
                torso_angle = calculate_torso_angle(shoulder_center, hip_center)

                # 姿势判断
                if 75 < torso_angle < 105:  # 理想侧平板角度范围
                    if not GOOD_POSTURE:
                        timer_start = time.time()
                        GOOD_POSTURE = True
                    duration = int(time.time() - timer_start)
                    cv2.putText(image, f"Time: {duration}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    GOOD_POSTURE = False
                    cv2.putText(image, "Adjust Posture!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")

        pose_estimator.draw_landmarks(image)
        cv2.imshow('Side Plank Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()