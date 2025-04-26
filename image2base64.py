import base64
import json

def image_to_base64_json(image_path, output_file="output.json"):
    """
    将指定路径的图片文件转换为 Base64 编码的字符串，
    并将结果以 JSON 格式保存到文件中。

    Args:
        image_path (str): 图片文件的路径。
        output_file (str, optional): 保存 JSON 数据的文件的路径。
                             默认为 "output.json"。

    Returns:
        bool: True 如果转换和保存成功，False 如果发生错误。
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        json_data = {"image": encoded_string}

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=4)  # 使用 indent 使 JSON 更易读

        print(f"Base64 encoded string saved to {output_file} in JSON format")
        return True

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# 示例用法
image_file = "img.png"  # 替换为你的图片文件路径
output_file = "image_base64.json"  # 你想要保存的 JSON 文件名
if image_to_base64_json(image_file, output_file):
    print("Conversion and save to JSON successful!")
else:
    print("Conversion or save failed.")