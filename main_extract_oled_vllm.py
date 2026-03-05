# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
# python main_tongyi_extract_img2json.py  --dir2process /mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/output/pdf_extract/run_test --skip_n 0 

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


def main(opt):
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # table_imgs_dir = os.path.join(output_dir, f'table-recognize-{timestamp}')
        table_imgs_dir = os.path.join(output_dir, output_dir_name)
        if not os.path.isdir(table_imgs_dir):
            print(f"{table_imgs_dir} 不存在跳过")
            continue
        # print("处理csv文件夹： ", table_imgs_dir)
        
        extract_csv_script = os.path.join(os.getcwd(), "tongyi_get_mol_dict_from_table_img.py")
        command = [
            "python", extract_csv_script,
             table_imgs_dir]
        subprocess.run(command, check=True)
        # raise
        


if __name__ == "__main__":
    args = get_run_args()

    main(opt=args)



