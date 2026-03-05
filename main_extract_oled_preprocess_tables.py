# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
# python main_extract_pdf_csv_only.py  --dir2process /mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/output/pdf_extract/run_test --pdf_dir /mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/data/pdf_examples/run_test --skip_n 0
import cv2
import argparse
from datetime import datetime
import os
import re
from pathlib import Path
from tqdm import tqdm
import subprocess

import difflib

from collections import defaultdict

table_extend_rate_h = 0.3
table_extend_rate_w = 0.15

output_dir_name = 'table-recognize-for-ai'
output_table_img_dir_name = 'table-images'

print(f"表格提取范围扩大因子h,w： {table_extend_rate_h, table_extend_rate_w}")


def boxes_overlap(box1, box2, threshold=0.1):
  """判断两个归一化框是否重叠（IoU > threshold）"""
  x1_min, y1_min, x1_max, y1_max = box1
  x2_min, y2_min, x2_max, y2_max = box2

  inter_x_min = max(x1_min, x2_min)
  inter_y_min = max(y1_min, y2_min)
  inter_x_max = min(x1_max, x2_max)
  inter_y_max = min(y1_max, y2_max)

  if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
    return False

  inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
  area1 = (x1_max - x1_min) * (y1_max - y1_min)
  area2 = (x2_max - x2_min) * (y2_max - y2_min)
  union_area = area1 + area2 - inter_area
  iou = inter_area / union_area if union_area > 0 else 0
  return iou > threshold


def merge_boxes(boxes):
  """简单合并重叠框：迭代合并直到无重叠"""
  merged = []
  boxes = sorted(boxes, key=lambda b: b[0])  # 按 x1 排序
  while boxes:
    base = boxes.pop(0)
    to_merge = [base]
    remaining = []
    for box in boxes:
      if boxes_overlap(base, box):
        to_merge.append(box)
      else:
        remaining.append(box)
    # 合并 to_merge 中的所有框
    x1 = min(b[0] for b in to_merge)
    y1 = min(b[1] for b in to_merge)
    x2 = max(b[2] for b in to_merge)
    y2 = max(b[3] for b in to_merge)
    merged.append((x1, y1, x2, y2))
    boxes = remaining
  return merged


def normalize_for_matching(name: str) -> str:
    """
  将原始 PDF 文件名标准化，便于和 folder_name 匹配：
  - 移除 .pdf
  - 替换多个空格为单个空格
  - 移除特殊字符（保留字母、数字、空格、中文、-）
  """
    name = os.path.splitext(name)[0]  # 去掉 .pdf
    # 保留中文、英文、数字、空格、连字符，其他替换为空格
    name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def folder_to_query(folder_name: str) -> str:
    """
  将规范化文件夹名转回“可匹配”的查询字符串：
  - 将 _ 替换为空格
  - 但保留 "-_" -> "-"
  """
    # 先把 "-_" 和 "_-" 还原为 "-"
    s = folder_name.replace("_-_", " - ")
    s = s.replace("_-", " -")
    s = s.replace("-_", "- ")
    # 再把剩余的 _ 替换为空格
    s = s.replace("_", " ")
    # 清理多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_matching_pdf(folder_name, normalized_pdfs, top_n=3) -> list:
    """
  在 pdf_dir 中查找与 folder_name 最匹配的 PDF 文件
  """
    query = folder_to_query(folder_name)
    print(f"🔍 查询字符串: '{query}'")

    # 获取所有 PDF 文件名（标准化后）

    # 提取标准化后的名称用于匹配
    norm_names = [item[1] for item in normalized_pdfs]

    # 使用 difflib 模糊匹配
    matches = difflib.get_close_matches(query, norm_names, n=top_n, cutoff=0.6)

    # 返回原始文件名
    matched_files = []
    for match in matches:
        for orig_file, norm in normalized_pdfs:
            if norm == match:
                matched_files.append(orig_file)
                break

    return matched_files


def find_latest_molecules_result_folder(base_dir):
    """
  在 base_dir 中查找以 'molecules_detect_results_' 开头且时间戳最新的文件夹

  :param base_dir: 要搜索的根目录路径
  :return: 最新文件夹的完整路径，若未找到则返回 None
  """
    pattern = re.compile(r'^molecules_detect_results_(\d{8}_\d{6})$')
    candidate_folders = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                timestamp_str = match.group(1)  # e.g., "20251120_080123"
                try:
                    # 将时间戳字符串解析为 datetime 对象，用于比较
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidate_folders.append((dt, item_path))
                except ValueError:
                    # 时间格式不合法，跳过
                    continue

    if not candidate_folders:
        return None

    # 按时间戳排序，取最新的（最大值）
    latest_folder = max(candidate_folders, key=lambda x: x[0])[1]
    return latest_folder


def get_run_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir2process', type=str, default='',
                      help="ALL PDF directories after csv and figures extraction.")
  parser.add_argument('--skip_n', type=int, default=0,
                      help="Resume process.")
  parser.add_argument('--pdf_dir', type=str, default='',
                      help="ALL PDF files.")
  parser.add_argument('--process_csv', action='store_true', help="ALL PDF files.")
  opt = parser.parse_args()
  return opt


def main(opt):
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n

    pdf_directory = opt.pdf_dir

    all_source_pdf_files = [f for f in os.listdir(opt.pdf_dir) if f.lower().endswith('.pdf')]

    print(f"处理文献文件夹共计数量{len(all_pdfs_dir)}")
    print(f"启动文件序号：{skip_i}")
    ii = 0
    for pdf_dir_name in tqdm(all_pdfs_dir):
        if ii < skip_i:
            ii += 1
            continue

        obj_dir = os.path.join(root_path, pdf_dir_name)
        detect_outputs = find_latest_molecules_result_folder(obj_dir)
        output_dir = os.path.join(root_path, pdf_dir_name, detect_outputs)

        pdf_names = [f for f in os.listdir(output_dir) if f.lower().endswith('.pdf')]
        if len(pdf_names) > 0:
            pdf_name = pdf_names[0]
            print(f"当前文件夹{pdf_dir_name}匹配的pdf名称为：{pdf_name}")
            if pdf_name not in all_source_pdf_files:
                print(f"当前文件夹{pdf_dir_name}匹配的pdf名称失败，未找到原始pdf文件")
                continue
        else:
            print(f"当前文件夹{pdf_dir_name}匹配的pdf名称失败，目标文件夹不包含pdf文件")
            continue

        pdf_path = os.path.join(opt.pdf_dir, pdf_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # csv_output_dir = os.path.join(output_dir, f'table-recognize-{timestamp}')
        csv_output_dir = os.path.join(output_dir, output_dir_name)
        os.makedirs(csv_output_dir, exist_ok=True)

        output_tables_crop_dir = os.path.join(output_dir, output_table_img_dir_name)
        os.makedirs(output_tables_crop_dir, exist_ok=True)

        cv_detection_labels_dir = os.path.join(root_path, pdf_dir_name, detect_outputs, 'detection_output', 'labels')
        page_tables = defaultdict(list)
        for label_file in Path(cv_detection_labels_dir).glob("*.txt"):
            image_name = label_file.stem
            with open(label_file, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
                    # print(cls_id, cx_norm, cy_norm, w_norm, h_norm )

                    if cls_id == 3:
                        # Tables
                        x1 = max(0, cx_norm - (w_norm + table_extend_rate_w) / 2)
                        y1 = max(0, cy_norm - (h_norm + table_extend_rate_h) / 2) 
                        x2 = min(1, cx_norm + (w_norm + table_extend_rate_w) / 2)
                        y2 = min(1, cy_norm + (h_norm + table_extend_rate_h) / 2)  # 后面可以多一些
                        # print(x1, y1, x2, y2)
                        page_id = image_name.split('_')[1]  # labels page_xx.txt  -> iamge_name page_xx
                        page_tables[page_id].append((x1, y1, x2, y2))
                        if opt.process_csv:
                          extract_csv_script = os.path.join(os.getcwd(), "get_csv_from_pdf_by_zone_without_save.py")
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
                              '--upleft',
                          ]
                          subprocess.run(command, check=True)
                          # raise
                    else:
                      # 其他类别不做处理
                      pass

        # Step 2: 对每个页面的表格区域进行合并
        merged_page_tables = {}
        for page_id, boxes in page_tables.items():
          merged = merge_boxes(boxes)
          merged_page_tables[page_id] = merged

        # Step 3: 读取 PNG 并裁剪
        for page_id, boxes in merged_page_tables.items():
          png_path = Path(obj_dir) / f"page_{page_id}.png"
          if not png_path.exists():
            print(f"Warning: {png_path} not found, skipping.")
            continue

          img = cv2.imread(str(png_path))
          h_img, w_img = img.shape[:2]

          for idx_, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(boxes):
            x1_px = int(x1_norm * w_img)
            y1_px = int(y1_norm * h_img)
            x2_px = int(x2_norm * w_img)
            y2_px = int(y2_norm * h_img)

            cropped = img[y1_px:y2_px, x1_px:x2_px]
            output_path = Path(output_tables_crop_dir) / f"page_{page_id}_table_{idx}.png"
            cv2.imwrite(str(output_path), cropped)

        print("✅ 所有表格区域已合并并裁剪保存。")
        # raise



if __name__ == "__main__":
    args = get_run_args()

    main(opt=args)



