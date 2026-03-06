# OLED-MAMA
🧪 OLED-MAMA is a modular, multi-agent pipeline designed to automatically extract structured molecular data from scientific PDFs related to Organic Light-Emitting Diodes (OLEDs). By combining computer vision, OCR, large language models (LLMs), and molecular recognition, it transforms unstructured literature into clean, machine-readable datasets ready for downstream machine learning tasks.

## 🛠️ Environment Requirements

Main dependencies (tested versions):
- easyocr                   1.7.2
- opencv-python             4.8.0.74
- pdf2image                 1.16.3
- dashscope                 1.25.2

## 🚀 Quick Start
Make the script executable and run:
chmod +x run_pipeline.sh
./run_pipeline.sh <raw_pdf_dir> <yolo_weight> <yolo_project> [output_dir] [gpu_id]
Example:
./run_pipeline.sh /data/pdf_examples/ yolov5l/best.pt YOLOv5 output/pdf_extract/run_test 0

<raw_pdf_dir> – directory containing the PDF files to process

<yolo_weight> – path to YOLOv5 weights (best.pt)

<yolo_project> – name of the YOLO project (used internally)

[output_dir] – optional output directory (default: ./output)

[gpu_id] – optional GPU ID for YOLO inference (default: CPU)

## 🔍 Pipeline Steps
If you prefer to run each agent separately, follow the steps below. All commands assume you are in the project root.
Preprocessing with YOLO

Step 1: Preprocess PDFs (Figure/Table Detection)
Step1 python main_extract_oled_preprocess.py --model_pt your_YOLO/weights/best.pt --yolo_project_path your_YOLO_master  --pdf_dir /data/pdf_examples/ --output_dir output/pdf_extract/run_tes
Step 2: Extract Tables images
Step2 python main_extract_pdf_csv_only.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0
Step 3: Extract Tables data
Step3 python main_tongyi_extract_img2json.py  --dir2process your_output_dir --skip_n 0 
Step 4: OCR Text - molecules mapping
Step4 python main_extract_oled_ocr.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0 --gpu 0

🧬Molecular Structure → SMILES Conversion
基于上述4个步骤的结果使用Molecule structure 进一步转换smiles 码。提取的分子名称-匹配图像结果会放置在xxx，提取的分子名称-属性Json结果会放置在table_image文件夹
运行model种的Molecule structure提取脚本，案例 python run_detect_cpu_for_pdf_dirs.py --ckpt 'MolScribe/model/swin_base_char_aux_1m.pth' --dir2process your_output_dir --skip_n 0

你可以根据自己的需求对agent的实际工作进行修改从而满足你的需求。

📊Downstream Machine Learning
The MLs/ folder contains six curated OLED material property datasets (TADF molecules) used in our paper. You can use these datasets directly or generate your own from extracted data.

To create a training dataset from your own extracted JSON files:
你可以用我们提供的 create_dataset_mol_get_properties.py 脚本从你自己提取的数据中提取数据训练你的机器学习模型。




