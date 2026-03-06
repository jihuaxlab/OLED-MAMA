# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
# python main_extract_oled_preprocess_table.py  --dir2process output/pdf_extract/run_test --pdf_dir /data/pdf_examples/run_test --skip_n 0
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

print(f"Table extraction expansion factors (h, w): {table_extend_rate_h, table_extend_rate_w}")


def boxes_overlap(box1, box2, threshold=0.1):
    """Check if two normalized bounding boxes overlap (IoU > threshold)"""
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
    """Simple iterative merging of overlapping boxes until no overlaps remain"""
    merged = []
    boxes = sorted(boxes, key=lambda b: b[0])  # Sort by x1
    while boxes:
        base = boxes.pop(0)
        to_merge = [base]
        remaining = []
        for box in boxes:
            if boxes_overlap(base, box):
                to_merge.append(box)
            else:
                remaining.append(box)
        # Merge all boxes in to_merge
        x1 = min(b[0] for b in to_merge)
        y1 = min(b[1] for b in to_merge)
        x2 = max(b[2] for b in to_merge)
        y2 = max(b[3] for b in to_merge)
        merged.append((x1, y1, x2, y2))
        boxes = remaining
    return merged


def normalize_for_matching(name: str) -> str:
    """
    Normalize raw PDF filenames for matching with folder names:
    - Remove .pdf extension
    - Replace multiple spaces with a single space
    - Keep only alphanumeric characters, Chinese characters, spaces, and hyphens; replace others with space
    """
    name = os.path.splitext(name)[0]  # Remove .pdf
    # Keep letters, digits, Chinese chars, spaces, and hyphens; replace others with space
    name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def folder_to_query(folder_name: str) -> str:
    """
    Convert a normalized folder name back into a query string for matching:
    - Replace underscores with spaces
    - But preserve "-_" patterns as "-"
    """
    # First restore "-_" and "_-" to "-"
    s = folder_name.replace("_-_", " - ")
    s = s.replace("_-", " -")
    s = s.replace("-_", "- ")
    # Then replace remaining underscores with spaces
    s = s.replace("_", " ")
    # Clean up extra whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_matching_pdf(folder_name, normalized_pdfs, top_n=3) -> list:
    """
    Find the most matching PDF file(s) in pdf_dir for a given folder name
    """
    query = folder_to_query(folder_name)
    print(f"🔍 Query string: '{query}'")

    # Extract normalized names for matching
    norm_names = [item[1] for item in normalized_pdfs]

    # Use difflib for fuzzy matching
    matches = difflib.get_close_matches(query, norm_names, n=top_n, cutoff=0.6)

    # Return original filenames
    matched_files = []
    for match in matches:
        for orig_file, norm in normalized_pdfs:
            if norm == match:
                matched_files.append(orig_file)
                break

    return matched_files


def find_latest_molecules_result_folder(base_dir):
    """
    Find the latest folder in base_dir that starts with 'molecules_detect_results_' followed by a timestamp.

    :param base_dir: Root directory to search
    :return: Full path to the latest folder, or None if not found
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
                    # Parse timestamp string into datetime object for comparison
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidate_folders.append((dt, item_path))
                except ValueError:
                    # Invalid timestamp format; skip
                    continue

    if not candidate_folders:
        return None

    # Sort by timestamp and return the newest
    latest_folder = max(candidate_folders, key=lambda x: x[0])[1]
    return latest_folder


def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir2process', type=str, default='',
                        help="Root directory containing processed PDF subfolders (after CSV/figure extraction).")
    parser.add_argument('--skip_n', type=int, default=0,
                        help="Number of folders to skip (for resuming processing).")
    parser.add_argument('--pdf_dir', type=str, default='',
                        help="Directory containing original PDF files.")
    parser.add_argument('--process_csv', action='store_true', help="Enable CSV extraction from detected table regions.")
    opt = parser.parse_args()
    return opt


def main(opt):
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n

    pdf_directory = opt.pdf_dir

    all_source_pdf_files = [f for f in os.listdir(opt.pdf_dir) if f.lower().endswith('.pdf')]

    print(f"Total number of document folders to process: {len(all_pdfs_dir)}")
    print(f"Starting from index: {skip_i}")
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
            print(f"PDF name matched for folder '{pdf_dir_name}': {pdf_name}")
            if pdf_name not in all_source_pdf_files:
                print(f"Failed to locate original PDF for folder '{pdf_dir_name}': {pdf_name} not found in source directory")
                continue
        else:
            print(f"Failed to match PDF for folder '{pdf_dir_name}': no PDF found in target directory")
            continue

        pdf_path = os.path.join(opt.pdf_dir, pdf_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

                    if cls_id == 3:
                        # Class ID 3 corresponds to tables
                        x1 = max(0, cx_norm - (w_norm + table_extend_rate_w) / 2)
                        y1 = max(0, cy_norm - (h_norm + table_extend_rate_h) / 2) 
                        x2 = min(1, cx_norm + (w_norm + table_extend_rate_w) / 2)
                        y2 = min(1, cy_norm + (h_norm + table_extend_rate_h) / 2)
                        page_id = image_name.split('_')[1]  # e.g., 'page_03' -> '03'
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
                    else:
                        # Ignore other classes
                        pass

        # Step 2: Merge overlapping table regions on each page
        merged_page_tables = {}
        for page_id, boxes in page_tables.items():
            merged = merge_boxes(boxes)
            merged_page_tables[page_id] = merged

        # Step 3: Load PNG pages and crop table regions
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

        print("✅ All table regions have been merged, cropped, and saved.")


if __name__ == "__main__":
    args = get_run_args()
    main(opt=args)
