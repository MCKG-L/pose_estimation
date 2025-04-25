import cv2
import mediapipe as mp
import numpy as np
import threading
from modules.pose_estimation import PoseEstimator
from modules.text_to_speech import speak

# 高抬腿动作
def detect_highknee():
    cap = cv2.VideoCapture(0)
    pose_estimator = PoseEstimator(min_tracking_confidence=0.7)
    mp_pose = mp.solutions.pose

    count = 0
    knee_heights = []
    HIP_Y = None

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

                # 初始化髋部基准高度
                if HIP_Y is None:
                    HIP_Y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2

                # 实时获取双膝高度
                left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                current_height = (HIP_Y - min(left_knee_y, right_knee_y))

                # 动态阈值判断（自动校准）
                knee_heights.append(current_height)
                if len(knee_heights) > 10:
                    threshold = np.mean(knee_heights[-10:]) * 1.5
                    if current_height > threshold:
                        count += 1
                        threading.Thread(target=speak, args=(f"高抬腿 {count} 次",)).start()
                        knee_heights = []  # 重置校准

                # 可视化
                cv2.putText(image, f"Count: {count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(image,
                         (0, int(HIP_Y * 480)),
                         (640, int(HIP_Y * 480)),
                         (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")

        pose_estimator.draw_landmarks(image)
        cv2.imshow('High Knee Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()