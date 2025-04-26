from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from modules.pose_estimation import PoseEstimator
from modules.angle_calculation import calculate_angle, calculate_torso_angle
import base64

app = Flask(__name__)
# CORS(app)
pose_estimator = PoseEstimator()

def detect_curl_analysis(landmarks):
    """分析弯举动作"""
    if not landmarks:
        return {"error": "No landmarks provided"}

    if landmarks and len(landmarks) > mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value:
        right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['x'],
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]['y']]
        right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]['x'],
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]['y']]
        right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]['x'],
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]['y']]

        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        feedback = []
        if elbow_angle > 110:
            feedback.append("Elbow angle too large. Keep your upper arm still.")
        elif elbow_angle < 60:
            feedback.append("Elbow angle too small. Ensure full range of motion.")

        return {"feedback": feedback, "elbow_angle": elbow_angle}
    else:
        return {"error": "Incomplete landmark data"}


def detect_pushup_analysis(landmarks):
    """分析俯卧撑动作"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value:
        return {"error": "Incomplete landmark data"}

    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['x'],
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['y']]
    left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]['x'],
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]['y']]
    left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]['x'],
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]['y']]

    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    feedback = []
    if elbow_angle > 160:
        feedback.append("手臂伸直")
    if elbow_angle < 90:
        feedback.append("手臂弯曲")

    return {"feedback": feedback, "elbow_angle": elbow_angle}


def detect_squat_analysis(landmarks):
    """分析深蹲动作"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_HIP.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_KNEE.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value:
        return {"error": "Incomplete landmark data"}

    left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y']]
    left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]['x'],
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]['y']]
    left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['x'],
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['y']]

    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    feedback = []
    if knee_angle > 170:
        feedback.append("腿伸直")
    if knee_angle < 90:
        feedback.append("下蹲")

    return {"feedback": feedback, "knee_angle": knee_angle}


def detect_lunge_analysis(landmarks):
    """分析弓步动作"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_HIP.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_KNEE.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value or len(landmarks) < mp.solutions.pose.PoseLandmark.RIGHT_HIP.value or len(landmarks) < mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value or len(landmarks) < mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value:
        return {"error": "Incomplete landmark data"}

    left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y']]
    left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]['x'],
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]['y']]
    left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['x'],
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['y']]

    right_hip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['x'],
                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]['y']]
    right_knee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]['x'],
                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]['y']]
    right_ankle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]['x'],
                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]['y']]

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    feedback = []
    if min(left_knee_angle, right_knee_angle) < 90:
        feedback.append("膝盖弯曲")
    if max(left_knee_angle, right_knee_angle) > 140:
        feedback.append("腿伸直")

    return {"feedback": feedback, "left_knee_angle": left_knee_angle, "right_knee_angle": right_knee_angle}


def detect_plank_analysis(landmarks):
    """分析平板支撑动作"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_HIP.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value:
        return {"error": "Incomplete landmark data"}

    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['x'],
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['y']]
    left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'],
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y']]
    left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['x'],
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['y']]

    torso_angle = calculate_torso_angle(left_shoulder, left_hip)
    hip_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

    feedback = []
    if not (160 < hip_angle < 190):
        feedback.append("抬臀或塌腰")
    if not (torso_angle < 10):
        feedback.append("身体未保持直线")

    return {"feedback": feedback, "torso_angle": torso_angle, "hip_angle": hip_angle}


def detect_highknee_analysis(landmarks):
    """分析高抬腿动作"""

    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_HIP.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_KNEE.value:
        return {"error": "Incomplete landmark data"}

    left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y']]
    left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]['x'],
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]['y']]

    height_difference = left_hip[1] - left_knee[1]  # 简化为y坐标的差值

    feedback = []
    if height_difference < 0.3:  # 需要根据实际情况调整阈值
        feedback.append("膝盖抬得不够高")
    else:
        feedback.append("膝盖抬高")

    return {"feedback": feedback, "height_difference": height_difference}


def detect_jump_analysis(landmarks):
    """分析跳跃动作"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value:
        return {"error": "Incomplete landmark data"}

    left_ankle_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['y']

    # 这里需要一个更复杂的逻辑来判断跳跃，例如检测y坐标的局部最小值和峰值
    # 为了简化，这里只返回踝关节的y坐标
    return {"feedback": ["Jump analysis in progress"], "left_ankle_y": left_ankle_y}


def detect_jumping_jack_analysis(landmarks):
    """分析开合跳动作"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value or len(landmarks) < mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value:
        return {"error": "Incomplete landmark data"}

    left_ankle_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]['x']
    right_ankle_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]['x']
    feet_distance = abs(right_ankle_x - left_ankle_x)

    feedback = []
    if feet_distance < 0.1:  # 需要根据实际情况调整阈值
        feedback.append("双脚并拢")
    else:
        feedback.append("双脚分开")

    return {"feedback": feedback, "feet_distance": feet_distance}


def detect_sideplank_analysis(landmarks):
    """分析侧平板支撑"""
    if not landmarks or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value or len(landmarks) < mp.solutions.pose.PoseLandmark.LEFT_HIP.value:
        return {"error": "Incomplete landmark data"}

    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['x'],
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]['y']]
    left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['x'],
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]['y']]

    torso_angle = calculate_torso_angle(left_shoulder, left_hip)

    feedback = []
    if not (80 < torso_angle < 100):  # 需要根据实际情况调整阈值范围
        feedback.append("身体未保持侧向直线")
    else:
        feedback.append("姿势正确")

    return {"feedback": feedback, "torso_angle": torso_angle}


def detect_situp_analysis(landmarks):
    return {"feedback": "Situp analysis not yet implemented"}

#
# def process_frame(frame_data):
#     """处理单帧图像进行姿态估计"""
#     try:
#         img_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
#         frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
#         if frame is None:
#             print("Failed to decode frame")
#             return None
#
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose_estimator.process(image)
#
#         landmarks = []
#         if results and results.pose_landmarks:
#             for landmark in results.pose_landmarks.landmark:
#                 landmarks.append({
#                     'x': landmark.x,
#                     'y': landmark.y,
#                     'z': landmark.z,
#                     'visibility': landmark.visibility
#                 })
#         return landmarks
#     except Exception as e:
#         print(f"Error processing frame: {e}")
#         return None
#
# from R import R  # 导入 R 类
# @app.route('/pose_estimation/<exercise>', methods=['POST'])
# def estimate_pose(exercise):
#     """接收图像数据和运动类型，返回姿态估计结果"""
#     try:
#         data = request.get_json()
#         frame_data = data.get('image')  # Base64编码的图像数据
#         if not frame_data:
#             return jsonify(R.error(400, "没有提供图像数据")), 400
#
#         landmarks = process_frame(frame_data)
#         if landmarks is None:
#             return jsonify(R.error(500, "姿态估计失败")), 500
#
#         # 根据运动类型调用相应的分析函数
#         exercise_results = {}
#         if exercise == 'curl':
#             exercise_results = detect_curl_analysis(landmarks)
#         elif exercise == 'pushup':
#             exercise_results = detect_pushup_analysis(landmarks)
#         elif exercise == 'squat':
#             exercise_results = detect_squat_analysis(landmarks)
#         elif exercise == 'lunge':
#             exercise_results = detect_lunge_analysis(landmarks)
#         elif exercise == 'plank':
#             exercise_results = detect_plank_analysis(landmarks)
#         elif exercise == 'highknee':
#             exercise_results = detect_highknee_analysis(landmarks)
#         elif exercise == 'jump':
#             exercise_results = detect_jump_analysis(landmarks)
#         elif exercise == 'jumping_jack':
#             exercise_results = detect_jumping_jack_analysis(landmarks)
#         elif exercise == 'sideplank':
#             exercise_results = detect_sideplank_analysis(landmarks)
#         elif exercise == 'situp':
#             exercise_results = detect_situp_analysis(landmarks)
#         else:
#             return jsonify(R.error(400, "无效的运动类型")), 400
#
#         # 返回姿态估计的结果
#         return jsonify(R.ok("姿态估计成功", {"landmarks": landmarks, "exercise_results": exercise_results})), 200
#
#     except Exception as e:
#         # 发生异常时返回错误信息
#         return jsonify(R.error(500, str(e))), 500

def process_frame(frame_data):
    """处理单帧图像进行姿态估计"""
    try:
        img_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode frame")
            return None, None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = []
        if results and results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            pose_estimator.draw_landmarks(image)

            # # 使用 mediapipe 绘制标记
            # mp_drawing = mp.solutions.drawing_utils
            # mp_pose = mp.solutions.pose
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image, landmarks
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, None


from R import R
@app.route('/pose_estimation/<exercise>', methods=['POST'])
def estimate_pose(exercise):
    """接收图像数据和运动类型，返回姿态估计结果"""
    try:
        data = request.get_json()
        frame_data = data.get('image')  # Base64编码的图像数据
        if not frame_data:
            return jsonify(R.error(400, "没有提供图像数据")), 400

        frame, landmarks = process_frame(frame_data)
        if landmarks is None or frame is None:
            return jsonify(R.error(500, "姿态估计失败")), 500

        # 根据运动类型调用相应的分析函数
        if exercise == 'curl':
            exercise_results = detect_curl_analysis(landmarks)
        elif exercise == 'pushup':
            exercise_results = detect_pushup_analysis(landmarks)
        elif exercise == 'squat':
            exercise_results = detect_squat_analysis(landmarks)
        elif exercise == 'lunge':
            exercise_results = detect_lunge_analysis(landmarks)
        elif exercise == 'plank':
            exercise_results = detect_plank_analysis(landmarks)
        elif exercise == 'highknee':
            exercise_results = detect_highknee_analysis(landmarks)
        elif exercise == 'jump':
            exercise_results = detect_jump_analysis(landmarks)
        elif exercise == 'jumping_jack':
            exercise_results = detect_jumping_jack_analysis(landmarks)
        elif exercise == 'sideplank':
            exercise_results = detect_sideplank_analysis(landmarks)
        elif exercise == 'situp':
            exercise_results = detect_situp_analysis(landmarks)
        else:
            return jsonify(R.error(400, "无效的运动类型")), 400

        print(exercise_results)
        # 根据exercise_results绘制反馈信息到frame
        y_offset = 50  # 反馈文本绘制的初始Y坐标
        for key, feedback in exercise_results.items():
            # 每个反馈项绘制到图像上
            cv2.putText(frame, f"{key}: {feedback}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30  # 每条信息之间的间隔

        # # 在frame上绘制运动分析的反馈
        # cv2.putText(frame, f"Exercise Results: {exercise_results}", (10, y_offset),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite("test_frame.jpg", frame)
        # 返回带有绘制结果的frame
        _, img_encoded = cv2.imencode('.jpg', frame)
        frame_data = img_encoded.tobytes()

        return jsonify(R.ok("姿态估计成功", {"exercise_results": exercise_results,
                                             "image": base64.b64encode(frame_data).decode('utf-8')})), 200

        # return jsonify(R.ok("姿态估计成功", {"landmarks": landmarks, "exercise_results": exercise_results})), 200

    except Exception as e:
        return jsonify(R.error(500, str(e))), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)