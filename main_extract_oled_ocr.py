# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
# python main_extract_oled_ocr.py  --dir2process output/pdf_extract/run_test --pdf_dir /data/pdf_examples/run_test --skip_n 0 --gpu 0
import cv2
import argparse
from datetime import datetime
import os
import re
import json
from tqdm import tqdm
import difflib
from collections import defaultdict
import tools
import shutil

from datetime import datetime
import easyocr

ocr_prob = 0.3

output_table_img_dir_name = 'table-images'
output_mol_img_dir_name = 'cropped_images'
final_mol_img_dir_name = 'extracted_images_fix'
final_mol_for_ai_dir_name = 'extracted_images_for_ai'
final_ocr_json = 'ocr_closest_to_center_fix.json'


def sanitize_filename(text, space_replacement='_'):
    # Remove illegal characters
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
    # Compress whitespace and replace spaces
    cleaned = ' '.join(cleaned.split())
    if space_replacement:
        cleaned = cleaned.replace(' ', space_replacement)
    # Avoid starting/ending with . or - (optional)
    cleaned = cleaned.strip('-.')
    # Avoid empty filenames
    if not cleaned:
        cleaned = "unnamed"
    return cleaned


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
    parser.add_argument('--gpu', type=int, default=0,
                        help="Single GPU for EasyOCR.")
    parser.add_argument('--pdf_dir', type=str, default='',
                        help="Directory containing original PDF files.")
    parser.add_argument('--process_csv', action='store_true', help="LLM extract info to JSON.")
    opt = parser.parse_args()
    return opt


def main(opt):
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n

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

        output_mol_crop_dir = os.path.join(output_dir, output_mol_img_dir_name)
        if not os.path.isdir(output_mol_crop_dir):
            print(f"Folder '{output_mol_crop_dir}' does not exist!")
            continue

        output_tables_crop_dir = os.path.join(output_dir, output_table_img_dir_name)
        if not os.path.isdir(output_tables_crop_dir):
            print(f"Folder '{output_tables_crop_dir}' does not exist!")
            continue

        final_mol_dir = os.path.join(output_dir, final_mol_img_dir_name)
        if os.path.exists(final_mol_dir):
            shutil.rmtree(final_mol_dir)
        os.mkdir(final_mol_dir)

        final_mol_for_ai_dir = os.path.join(output_dir, final_mol_for_ai_dir_name)
        if os.path.exists(final_mol_for_ai_dir):
            shutil.rmtree(final_mol_for_ai_dir)
        os.mkdir(final_mol_for_ai_dir)

        output_temp_dir = os.path.join(output_dir, 'tmp')
        if not os.path.exists(output_temp_dir):
            os.mkdir(output_temp_dir)

        all_mol_keys = set()
        json_files = [f for f in os.listdir(output_tables_crop_dir) if f.endswith('.json')]
        for jf in json_files:
            with open(os.path.join(output_tables_crop_dir, jf), 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if 'All_Mol' in data and isinstance(data['All_Mol'], dict):
                        all_mol_keys.update(data['All_Mol'].keys())
                except Exception as e:
                    print(f"⚠️ JSON parsing failed for {jf}: {e}")
                    continue

        if not all_mol_keys:
            print(f"⚠️ No molecule names (All_Mol) found in current folder {pdf_dir_name}")
            continue

        print(f"🔍 Found {len(all_mol_keys)} candidate molecule names: {sorted(list(all_mol_keys))}")
        key_patterns = tools.build_core_patterns_from_examples(list(all_mol_keys))
        print('key_patterns:', key_patterns)

        # Step 2: Group molecular images by page
        mol_images = [f for f in os.listdir(output_mol_crop_dir) if f.endswith('.jpg')]
        page_groups = defaultdict(list)
        for img in mol_images:
            # File name format: page_10_crop0_cls0.jpg
            match = re.search(r'page_(\d+)_crop\d+_cls\d+\.jpg', img)
            if match:
                page_id = int(match.group(1))
                page_groups[page_id].append(img)
            else:
                print(f"⚠️ Unable to parse image page number: {img}")
        if len(page_groups) == 0:
            print(f"⚠️ No valid image page numbers found")
            continue

        # Step 3: Run OCR on each molecular image and match candidate names
        filtered_valid_keys_dict = defaultdict(list)
        for page_id, img_list in page_groups.items():
            valid_keys_dict = defaultdict(list)  # To record all data for the page
            directions = {'bottom': 0, 'top': 0, 'right': 0, 'left': 0}
            for img_file in img_list:
                img_path = os.path.join(output_mol_crop_dir, img_file)
                try:
                    image = cv2.imread(img_path)
                    h, w = image.shape[:2]
                    print("Processing", img_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    gray_img_path = os.path.join(output_temp_dir, f'gray_captcha_{timestamp}.png')
                    cv2.imwrite(gray_img_path, gray)
                    reader = easyocr.Reader(
                        ['en'],
                        model_storage_directory="/home/hezhaoming/.EasyOCR/model",
                        gpu=True)
                    results = reader.readtext(gray_img_path, detail=1)

                    if not results or not results[0]:
                        continue
                    boxes_and_txts = []
                    for (bbox, text, prob) in results:
                        if prob < ocr_prob:
                            continue
                        if not text:
                            continue
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        boxes_and_txts.append((text, (x1, y1, x2, y2)))

                    img_center = (w / 2, h / 2)
                    min_distance = 999999
                    closest_text_center = None
                    closest_text = None
                    for txt, bbox in boxes_and_txts:
                        clean_txt = txt.strip()
                        if len(clean_txt) == 0:
                            continue
                        matched_key = False
                        if clean_txt in all_mol_keys:
                            matched_key = clean_txt
                        else:
                            matched_key, matched_list = tools.is_candidate_valid(txt, key_patterns, max_extra=2)
                            if matched_key:
                                matched_key = ' '.join(matched_list)
                            else:
                                pass

                        if matched_key:
                            x1, y1, x2, y2 = bbox
                            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            dist = ((bbox_center[0] - img_center[0]) ** 2 + (
                                        bbox_center[1] - img_center[1]) ** 2) ** 0.5
                            valid_keys_dict[matched_key].append({
                                'bbox': [x1, y1, x2, y2],
                                'bbox_center': bbox_center,
                                'img_wh': [w, h],
                                'bbox_ref_distance': round(dist, 2),
                                'page_id': page_id,
                                'img_file_name': img_file
                            })
                            if dist < min_distance:
                                closest_text_center = bbox_center
                                closest_text = matched_key

                    if closest_text_center is not None:
                        center_y = closest_text_center[1]
                        center_x = closest_text_center[0]
                        if center_y < h * 0.35:
                            directions['top'] += 1
                            print(f"closest_text {closest_text} up")
                        if center_y > h * 0.65:
                            directions['bottom'] += 1
                            print(f"closest_text {closest_text} bottom")
                        if center_x < w * 0.35:
                            directions['left'] += 1
                            print(f"closest_text {closest_text} left")
                        if center_x > w * 0.65:
                            directions['right'] += 1
                            print(f"closest_text {closest_text} right")
                except Exception as e:
                    print(f"⚠️ OCR failed for {img_path}: {e}")
                    continue

            dominant_dir = max(directions, key=directions.get)
            print(f"✅ Dominant annotation direction: {dominant_dir} (scores: {directions})")

            # Step 5: Filter out matches not aligned with the dominant direction
            for key, matches in valid_keys_dict.items():
                for m in matches:
                    w, h = m['img_wh']
                    center_x, center_y = m['bbox_center']
                    keep = False
                    if dominant_dir == 'top' and center_y < h * 0.4:
                        keep = True
                    elif dominant_dir == 'bottom' and center_y > h * 0.6:
                        keep = True
                    elif dominant_dir == 'left' and center_x < w * 0.4:
                        keep = True
                    elif dominant_dir == 'right' and center_x > w * 0.6:
                        keep = True
                    if keep:
                        filtered_valid_keys_dict[key].append(m)

        if len(filtered_valid_keys_dict) == 0:
            print(f"⚠️ No matching molecule names found via OCR in any molecular images")
        else:
            # Based on filtered_valid_keys_dict, merge images corresponding to keys
            final_mol_dir = os.path.join(output_dir, final_mol_img_dir_name)
            if not os.path.isdir(final_mol_dir):
                os.mkdir(final_mol_dir)

            final_json_path = os.path.join(output_dir, final_ocr_json)
            results = {}
            for key, matches in filtered_valid_keys_dict.items():
                sorted_matches = sorted(matches, key=lambda x: x['bbox_ref_distance'])[:2]
                if not sorted_matches:
                    continue

                images_to_merge = []
                image_paths = []
                image_infos = []
                distance_infos = []
                loaded_img = []
                for match in sorted_matches:
                    img_path = os.path.join(output_mol_crop_dir, match['img_file_name'])
                    if img_path in loaded_img:
                        continue
                    image = cv2.imread(img_path)
                    loaded_img.append(img_path)
                    if image is None:
                        print(f"Warning: Unable to read image {img_path}")
                        continue
                    images_to_merge.append(image)
                    image_paths.append(f"{final_mol_img_dir_name}/keys.jpg")  # Placeholder for merged image path
                    image_infos.append(match['img_file_name'])  # Original image filename
                    distance_infos.append(match['bbox_ref_distance'])

                if images_to_merge:
                    max_width = max([img.shape[1] for img in images_to_merge])
                    padded_images = []
                    for img in images_to_merge:
                        h, w = img.shape[:2]
                        pad_left = (max_width - w) // 2
                        pad_right = max_width - w - pad_left
                        padded = cv2.copyMakeBorder(
                            img,
                            top=0, bottom=5,
                            left=pad_left, right=pad_right,
                            borderType=cv2.BORDER_CONSTANT,
                            value=(0, 0, 255)
                        )
                        padded_images.append(padded)

                    merged_image = cv2.vconcat(padded_images)
                    merged_image_key = sanitize_filename(key)
                    merged_image_name = f"{merged_image_key}.jpg"
                    merged_image_path = os.path.join(final_mol_dir, merged_image_name)
                    cv2.imwrite(merged_image_path, merged_image)

                    closest_match = sorted_matches[0]
                    closest_img_path = os.path.join(output_mol_crop_dir, closest_match['img_file_name'])
                    ai_img_dest_path = os.path.join(final_mol_for_ai_dir, closest_match['img_file_name'])

                    try:
                        shutil.copy(closest_img_path, ai_img_dest_path)
                    except Exception as e:
                        print(f"❌ Error copying the closest molecule image: {e}")

                    results[key] = {
                        "distance": distance_infos,
                        "image_path": [f"{final_mol_img_dir_name}/{merged_image_name}"],  # Merged image path
                        "image_info": image_infos,  # Original image filenames
                    }

            final_json_path = os.path.join(output_dir, final_ocr_json)
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            print(f"✅ Results saved to {final_json_path}")


if __name__ == "__main__":
    args = get_run_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # EasyOCR uses GPU 0
    main(opt=args)

