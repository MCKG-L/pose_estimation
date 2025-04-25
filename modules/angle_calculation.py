import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def calculate_torso_angle(shoulder, hip):
    """计算躯干与垂直方向的夹角"""
    vertical = (0, -1)  # 垂直方向向量
    torso_vector = (hip[0]-shoulder[0], hip[1]-shoulder[1])
    dot_product = vertical[0]*torso_vector[0] + vertical[1]*torso_vector[1]
    magnitude = np.linalg.norm(vertical) * np.linalg.norm(torso_vector)
    angle = np.degrees(np.arccos(dot_product/magnitude))
    return angle

