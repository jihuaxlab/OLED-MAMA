# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
# python main_extract_oled_vllm.py  --dir2process output/pdf_extract/run_test --skip_n 0 

import argparse
from datetime import datetime
import os
import re
from pathlib import Path
from tqdm import  tqdm
import subprocess

import difflib


output_dir_name = 'table-images'


def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir2process', type=str, default='',
                        help="ALL PDF directories after csv and figures extraction.")
    parser.add_argument('--skip_n', type=int, default=0,
                        help="Resume process.")
    opt = parser.parse_args()
    return opt


def find_latest_molecules_result_folder(base_dir):
    """
    Search for the folder in the base_dir that starts with 'molecules_detect_results_' and has the latest timestamp. 

    :param base_dir: The root directory path to search
    :return: The complete path of the latest folder. If not found, return None
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
                    # Parse the timestamp string into a datetime object for comparison purposes.
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidate_folders.append((dt, item_path))
                except ValueError:
                    # The time format is invalid. Skipping.
                    continue

    if not candidate_folders:
        return None

    # Sort by timestamp and select the latest (maximum value)
    latest_folder = max(candidate_folders, key=lambda x: x[0])[1]
    return latest_folder


def main(opt):
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n

    print(f"The total number of document folders {len(all_pdfs_dir)}")
    print(f"Startup file number：{skip_i}")
    ii = 0
    for pdf_dir_name in tqdm(all_pdfs_dir):
        if ii < skip_i:
            ii += 1
            continue

        obj_dir = os.path.join(root_path, pdf_dir_name)
        detect_outputs = find_latest_molecules_result_folder(obj_dir)
        output_dir = os.path.join(root_path, pdf_dir_name, detect_outputs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # table_imgs_dir = os.path.join(output_dir, f'table-recognize-{timestamp}')
        table_imgs_dir = os.path.join(output_dir, output_dir_name)
        if not os.path.isdir(table_imgs_dir):
            print(f"{table_imgs_dir} not found")
            continue
        
        extract_csv_script = os.path.join(os.getcwd(), "tongyi_get_mol_dict_from_table_img.py")
        command = [
            "python", extract_csv_script,
             table_imgs_dir]
        subprocess.run(command, check=True)
        # raise
        


if __name__ == "__main__":
    args = get_run_args()

    main(opt=args)



