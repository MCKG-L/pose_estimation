from modules.exercise_curl import detect_curl
from modules.exercise_pushup import detect_pushup
from modules.exercise_situp import detect_situp
from modules.exercise_squat import detect_squat
from modules.exercise_plank import detect_plank
from modules.exercise_jump import detect_jump
from modules.exercise_lunge import detect_lunge
from modules.exercise_highknee import detect_highknee
from modules.exercise_sideplank import detect_sideplank
if __name__ == "__main__":
    print("请选择检测模式：")
    print("1. 弯举检测")
    print("2. 仰卧起坐")
    print("3. 俯卧撑")
    print("4. 深蹲检测")
    print("5. 平板支撑")
    print("6. 跳跃检测")
    print("7. 弓步检测")
    print("8. 高抬腿检测")
    print("9. 侧平板支撑")
    mode = input("请输入模式编号（1-9）：")

    if mode == '1':
        detect_curl()
    elif mode == '2':
        detect_situp()
    elif mode == '3':
        detect_pushup()
    elif mode == '4':
        detect_squat()
    elif mode == '5':
        detect_plank()
    elif mode == '6':
        detect_jump()
    elif mode == '7':
        detect_lunge()
    elif mode == '8':
        detect_highknee()
    elif mode == '9':
        detect_sideplank()
    else:
        print("无效输入，请输入 1-9 之间的数字。")
