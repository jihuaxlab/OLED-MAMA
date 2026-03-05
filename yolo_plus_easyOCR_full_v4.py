import subprocess
import os
from pathlib import Path
import cv2
import easyocr
from pdf2image import convert_from_path
from tqdm import tqdm
import numpy as np
import numpy as np
from collections import defaultdict
import json
import re


def sanitize_folder_name(name: str, max_length: int = 100) -> str:
    """
    清理文件夹名称，移除非法字符并限制长度
    :param name: 原始名称
    :param max_length: 最大长度（默认 100）
    :return: 安全的文件夹名
    """
    # 移除或替换 Windows / Linux / macOS 下的非法字符
    # 保留字母、数字、空格、连字符、下划线、点（但点不能开头/结尾）
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\[\],，。]', '_', name)  # 替换控制字符和常见非法字符为下划线
    name = re.sub(r'\s+', '_', name)                    # 多个空格/空白 → 单个下划线
    name = re.sub(r'_+', '_', name)                     # 多个下划线 → 单个
    name = name.strip('._ ')                            # 去掉首尾的 . _ 空格

    # 限制长度（避免路径过长）
    if len(name) > max_length:
        name = name[:max_length].rstrip('._ ')

    # 防止名字为空（极端情况）
    if not name:
        name = "unnamed_pdf"

    return name

    
# 将pdf转换为图像
def pdf_to_images(pdf_path, output_folder=None, dpi=200, first_page=None, last_page=None, output_name_max_len=50):
    """
    将 PDF 文件转换为图像列表
    :param pdf_path: PDF 文件路径
    :param output_folder: 输出图像保存路径（可选）
    :param dpi: 图像清晰度，默认 200
    :return: 图像对象列表
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
    except Exception as e:
        print(f"❌ 转换失败: {pdf_path} - 错误: {e}")
        return [], None

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        safe_pdf_name = sanitize_folder_name(pdf_name, output_name_max_len)
        pdf_output_folder = os.path.join(output_folder, safe_pdf_name)
        os.makedirs(pdf_output_folder, exist_ok=True)

        # 使用 tqdm 显示每页转换进度
        for i, image in enumerate(tqdm(images, desc=f"转换 {pdf_name}", leave=False)):
            image.save(os.path.join(pdf_output_folder, f"page_{i + 1}.png"), "PNG")

        print(f"输出图像到：{pdf_output_folder}")

    return images, pdf_output_folder


# 开启子线程用训练好的YOLOv5进行训练
def run_detect_and_crop_v5(image_folder, pdf_path, output_folder, weights, yolo_project_dir, expand_ratio=1.5):
    # 1. 调用 detect.py
    detect_script = os.path.join(yolo_project_dir, "detect.py")
    command = [
        "python", detect_script,
        "--weights", weights,
        "--source", image_folder,
        "--project", output_folder,
        "--imgsz", "1600",
        "--conf-thres", "0.35",
        "--iou-thres", "0.4",
        "--name", "detection_output",
        "--save-txt",  # 必须保存标签
        "--exist-ok"
    ]
    subprocess.run(command, check=True)

    # 2. 解析输出的labels和images
    detect_output_dir = Path(output_folder) / "detection_output"
    labels_dir = detect_output_dir / "labels"
    images_dir = Path(image_folder)

    csv_output_dir = os.path.join(output_folder, f'table-recognize')
    os.makedirs(csv_output_dir, exist_ok=True)

    crop_output_dir = Path(output_folder) / "cropped_images"
    crop_output_dir.mkdir(exist_ok=True)

    img_extensions = ['.png'] # 设置支持png
    # img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for label_file in labels_dir.glob("*.txt"):
        image_name = label_file.stem
        image_path = None
        for ext in img_extensions:
            img_p = images_dir / (image_name + ext)
            # print(f"检查图像路径{img_p}")
            if img_p.exists():
                image_path = img_p
                break

        if not image_path:
            continue

        print(f"处理图像{image_path}")

        img = cv2.imread(str(image_path))
        h, w, _ = img.shape

        with open(label_file, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
                # print(cx_norm, cy_norm, w_norm, h_norm )

                # 转换为像素坐标
                cx = int(cx_norm * w)
                cy = int(cy_norm * h)
                bbox_w = int(w_norm * w)
                bbox_h = int(h_norm * h)
                
                if cls_id == 0:
                    # 对于分子图像扩大bbox，对图注进行覆盖
                    new_w = int(bbox_w * expand_ratio)
                    new_h = int(bbox_h * expand_ratio)

                    x1 = max(0, cx - new_w // 2)
                    y1 = max(0, cy - new_h // 2)
                    x2 = min(w, cx + new_w // 2)
                    y2 = min(h, cy + new_h // 2)

                    cropped = img[y1:y2, x1:x2]
                    crop_save_path = crop_output_dir / f"{image_name}_crop{idx}_cls{cls_id}.jpg"
                    corp_img_info_dict = {'source': str(image_path), 
                                          'source_wh': [w, h],
                                          'bbox_raw_cxy_wh': [cx_norm, cy_norm, w_norm, h_norm],
                                          'bbox_expand_ratio': expand_ratio}
                    corp_img_info_json = crop_output_dir / f"{image_name}_crop{idx}_cls{cls_id}.json"
                    with open(corp_img_info_json, 'w', encoding='utf-8') as f:
                        json.dump(corp_img_info_dict, f, ensure_ascii=False, indent=4)
                    cv2.imwrite(str(crop_save_path), cropped)
                
                elif cls_id == 3:
                    # Tables
                    x1 = max(0, cx_norm - w_norm / 2)
                    y1 = max(0, cy_norm - h_norm / 2)
                    x2 = min(1, cx_norm + w_norm / 2)
                    y2 = min(1, cy_norm + h_norm / 2)
                    # print(x1, y1, x2, y2)
                    page_id = image_name.split('_')[1]  # labels page_xx.txt  -> iamge_name page_xx
                    extract_csv_script = os.path.join(os.getcwd(), "get_csv_from_pdf_by_zone_without_save.py")
                    pdf_name = Path(pdf_path).name
                    with open(os.path.join(output_folder, f'{pdf_name}'), 'w') as f_null:
                        f_null.write('\n')
                        f_null.close()
                    command = [
                            "python", extract_csv_script,
                            "-i", pdf_path,
                            "-o", csv_output_dir,
                            "--csv_name", f"tabele-in-page{page_id}_crop{idx}.csv",
                            "-p", page_id,
                            "--xn_start", str(x1),
                            "--yn_start", str(y1),
                            "--xn_end", str(x2),
                            "--yn_end", str(y2),
                            '--upleft',]
                    subprocess.run(command, check=True)
                    # raise
                else:
                    # 其他类别不做处理
                    pass


    print("YOLOv5 detection and cropping completed.")


# 使用示例
if __name__ == "__main__":

    pdf_2_extract_path = f"/mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/data/example.pdf"
#    obj_dir = "./data/pdf_images"
    output_dir = "./output/pdf_extract"
    model_pt = r"/mnt/LargeStorageSpace/HEZhaoming/YOLO/yolov5-master/trained_models/mol_recognize_v2/weights/best.pt"  # 模型
    yolo_project_path = r'/mnt/LargeStorageSpace/HEZhaoming/YOLO/yolov5-master'

    # start_page = 8
    # end_page = 9

    start_page = None
    end_page = None

    _, pdf_images_dir = pdf_to_images(pdf_path=pdf_2_extract_path, output_folder=output_dir, dpi=200, first_page=start_page, last_page=end_page)

    # pdf_images_dir = r"/mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/output/pdf_extract/example"

    molecules_detect_results_dir = os.path.join(pdf_images_dir, "molecules_detect_results")

    run_detect_and_crop_v5(pdf_images_dir, pdf_2_extract_path, output_folder=molecules_detect_results_dir, weights=model_pt, yolo_project_dir=yolo_project_path)

    # raise
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用 GPU 0

    cropped_images_dir = os.path.join(molecules_detect_results_dir,  "cropped_images")
    cropped_img_extensions = ['.jpg']
    all_images = [
        image_name for image_name in os.listdir(cropped_images_dir)
        if os.path.isfile(os.path.join(cropped_images_dir, image_name)) 
        and os.path.splitext(image_name)[1].lower() in cropped_img_extensions]
    # 使用 defaultdict 记录每个文本到图像中心的最短距离
    text_to_closest_mol_info = defaultdict(lambda: {'distance': float('inf'), 'image_path': None})  # 初始化为正无穷大
    for image_name in all_images:
        image_path = os.path.join(cropped_images_dir, image_name)
        print(f"解析 Corpped：{image_path}")
        image = cv2.imread(image_path)
        h, w = image.shape[:2] 

        page_infos = image_name.split('_')

        out_put_dir = r'./output/log' 
        print("处理",image_path )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img_path  = os.path.join(out_put_dir, 'gray_captcha.png')
        cv2.imwrite(gray_img_path, gray)

        reader = easyocr.Reader(
            [ 'en'],
            model_storage_directory="/home/hezhaoming/.EasyOCR/model",
            gpu=True
        )

        # 'ch_sim' 模型对 T vs J 区分能力弱 字符粘连或断裂	- 和 3 粘在一起，或 T 的横线缺失

        results = reader.readtext(gray_img_path, detail=1)

        # print("识别结果:\n", ''.join(results))

        for (bbox, text, prob) in results:
            print(f"文本: {text}")
            print(f"置信度: {prob:.4f}")
            print(f"位置框: {bbox}")  # bbox 是一个包含 4 个点的列表
            top_left = np.array(bbox[0]) - np.array([w/2, h/2])
            bottom_right = np.array(bbox[2]) - np.array([w/2, h/2])
            d_ocr_result_2_center = np.linalg.norm((top_left + bottom_right)/2)
            print(top_left, bottom_right)
            if d_ocr_result_2_center < text_to_closest_mol_info[text]['distance']:
                text_to_closest_mol_info[text]['distance'] = d_ocr_result_2_center
                text_to_closest_mol_info[text]['image_path'] = image_path
            print("-" * 40)

    print("*" * 40)
    for text, distance in text_to_closest_mol_info.items():
        print(f"文本 '{text}' 到图像中心的最短距离是: {distance}")
        
    result_dict = {
    text: {
        'distance': info['distance'],
        'image_path': info['image_path']
    }
    for text, info in text_to_closest_mol_info.items()
    }

    # === 保存为 JSON 文件 ===
    output_json_path = os.path.join(pdf_images_dir, "ocr_closest_to_center.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 处理完成！结果已保存至: {output_json_path}")

    
