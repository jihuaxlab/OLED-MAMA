# OLED-MAMA
**A multi-agent framework for automated extraction of OLED material data from scientific literature.**  
🧪 OLED-MAMA combines computer vision, OCR, and large language models to extract structured information (molecular structures, properties, tables) from PDF documents. It transforms unstructured literature into clean, machine-readable datasets ready for downstream machine learning tasks, enabling accelerated materials discovery.

[![Python 3.9+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7.2-green)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red)](https://opencv.org/)

---

## 🛠️ Environment Requirements

Main dependencies (tested versions):
- easyocr                   1.7.2
- opencv-python             4.8.0.74
- pdf2image                 1.16.3
- dashscope                 1.25.2

## 🚀 Quick Start

The entire pipeline can be launched with a single bash script:

```bash
./run_pipeline.sh <raw_pdf_dir> <yolo_weight> <yolo_project> [output_dir] [gpu_id]
```

**Example:**
```bash
./run_pipeline.sh /data/pdf_examples/ yolov5l/best.pt YOLOv5 output/pdf_extract/run_test 0
```

- `<raw_pdf_dir>` – directory containing the PDF files to process  
- `<yolo_weight>` – path to YOLOv5 weights (`best.pt`)  
- `<yolo_project>` – name of the YOLO project (used internally)  
- `[output_dir]` – optional output directory (default: `./output`)  
- `[gpu_id]` – optional GPU ID for YOLO inference

After execution, extracted table data (JSON) will be placed inside the `table_image` folder under your output directory.

---

## 🔍 Pipeline Steps
If you prefer to run each agent separately, follow the steps below. All commands assume you are in the project root. 用该步骤能够进一步提取分子结构

### Step 1: Preprocess PDFs (Figure/Table Detection)
python main_extract_oled_preprocess.py --model_pt your_YOLO/weights/best.pt --yolo_project_path your_YOLO_master  --pdf_dir /data/pdf_examples/ --output_dir output/pdf_extract/run_tes
### Step 2: Extract Tables images
python main_extract_pdf_csv_only.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0
### Step 3: Extract Tables data
python main_tongyi_extract_img2json.py  --dir2process your_output_dir --skip_n 0 
### Step 4: OCR Text - molecules mapping
python main_extract_oled_ocr.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0 --gpu 0

### 🧬Molecular Structure → SMILES Conversion
基于上述4个步骤的结果（也就是quick start结果）使用Molecule structure 进一步转换smiles 码。提取的分子名称-匹配图像结果会放置在xxx，提取的分子名称-属性Json结果会放置在table_image文件夹
运行model种的Molecule structure提取脚本，案例 python run_detect_cpu_for_pdf_dirs.py --ckpt 'MolScribe/model/swin_base_char_aux_1m.pth' --dir2process your_output_dir --skip_n 0

你可以根据自己的需求对agent的实际工作进行修改从而满足你的需求。

---

### 📁 Output Structure

A typical output directory (`output/pdf_extract/run_test`) looks like:

```
molecules_detect_results_xxxxxxxxxx/
├── table_image/           # extracted tables and JSON files for them
├── cropped_images/       # cropped molecular structure images
├── detection_output/     # raw YOLO outputs
├── table-recognize/      # CSV versions of tables by direct pdf conversion
├── ocr_closest_to_center_fix.json         # OCR mapping output
└── extracted_images_fix/                  # OCR output
```

---


## 📊Downstream Machine Learning
The MLs/ folder contains six curated OLED material property datasets (TADF molecules) used in our paper. You can use these datasets directly or generate your own from extracted data.

To create a training dataset from your own extracted JSON files:
你可以用我们提供的 create_dataset_mol_get_properties.py 脚本从你自己提取的数据中提取数据训练你的机器学习模型。


## 📚 Citation

If you use OLED-MAMA or our results in your research, please cite our paper (to be added).

---

*Happy extracting!* 🧪🔍




