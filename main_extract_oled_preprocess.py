import subprocess
import os
import yolo_plus_easyOCR_full_v4 as cv_lib
from collections import defaultdict
import cv2
import numpy as np
import easyocr
from tqdm import tqdm
import json
import shutil
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用 GPU 0
ocr_prob = 0.30 
ocr_expand_rate = 1.5 #截取分子时候扩大范围用于保存分子名称标注

table_extend_rate_h = 0.3
table_extend_rate_w = 0.15

model_pt = r"/mnt/LargeStorageSpace/HEZhaoming/YOLO/yolov5-master/trained_models/mol_recognize_v2/weights/best.pt" # 模型
yolo_project_path = r'/mnt/LargeStorageSpace/HEZhaoming/YOLO/yolov5-master'
pass_list = []

PDF_DIR = r"/mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/data/pdf_examples/run_test" # your pdf dir
output_dir = r'/mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/output/pdf_extract/run_test' # output dir

if not os.path.isdir(PDF_DIR):
	print('PDF文件夹不存在')
	sys.exit(1)

all_files = os.listdir(PDF_DIR) 
file_count = len(all_files)
ii = 0
resume_i = 0

if resume_i > 0:
    print("进度加载： resume_i=", resume_i)
for pdf_file_name in all_files:
    if ii < resume_i:
        ii +=1
        continue
    print("-" * 40)
    print("-" * 40)
    print(f"处理文件进度：{ii} // {file_count} ")
    print("-" * 40)
    print("-" * 40)
    if not pdf_file_name.lower().endswith('.pdf'):
        continue

    pdf_2_extract_path = os.path.join(PDF_DIR, pdf_file_name)
    start_page = None
    end_page = None

    _, pdf_images_dir = cv_lib.pdf_to_images(pdf_path=pdf_2_extract_path, output_folder=output_dir, dpi=200,
                                             first_page=start_page, last_page=end_page, output_name_max_len=50 )
    if None is pdf_images_dir:
        print(f"跳过文件：{pdf_2_extract_path}")
        continue
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    molecules_detect_results_dir = os.path.join(pdf_images_dir, f"molecules_detect_results_{timestamp}")

    cv_lib.run_detect_and_crop_v5(pdf_images_dir, pdf_2_extract_path, output_folder=molecules_detect_results_dir,
                                  weights=model_pt, yolo_project_dir=yolo_project_path, expand_ratio=ocr_expand_rate)
    # 不同于老版本的run_detect_and_crop_v5，此处要传入pdf_2_extract_path用来提取和处理csv
    
    cropped_images_dir = os.path.join(molecules_detect_results_dir, "cropped_images")
    extracted_image_dir = os.path.join(molecules_detect_results_dir,  "extracted_images") # 用于存放最终图像
    os.makedirs(extracted_image_dir, exist_ok=True)
    cropped_img_extensions = ['.jpg']
    all_images = [image_name for image_name in os.listdir(cropped_images_dir)
                  if os.path.isfile(os.path.join(cropped_images_dir, image_name))
                  and os.path.splitext(image_name)[1].lower() in cropped_img_extensions]

    # LLM提取表格数据
    # 概率匹配
    ii += 1



