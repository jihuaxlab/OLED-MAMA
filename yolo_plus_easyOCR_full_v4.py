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




    
