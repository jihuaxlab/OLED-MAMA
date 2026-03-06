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

当然可以！以下是完全使用标准 Markdown 语法（包含 #、##、### 等标题符号）编写的 纯文本版 README 内容，你可以直接复制粘贴到你的 README.md 文件中，GitHub 会正确渲染所有格式：

OLED-MAMA: Multi-Agent Framework for OLED Material Data Extraction

OLED-MAMA is a modular, multi-agent pipeline designed to automatically extract structured molecular data from scientific PDFs related to Organic Light-Emitting Diodes (OLEDs).

# OLED-MAMA 🔬✨

**A multi-agent framework for automated extraction of OLED material data from scientific literature.**  
OLED-MAMA combines computer vision, OCR, and large language models to extract structured information (molecular structures, properties, tables) from PDF documents, enabling accelerated materials discovery.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7.2-green)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red)](https://opencv.org/)

---

## 📖 Overview

OLED-MAMA is a modular, agent‑based pipeline designed to extract molecular data from OLED‑related research papers. It processes PDF collections through four sequential agents:

1. **Preprocessing Agent** – detects figures/tables using YOLO  
2. **Table Extraction Agent** – converts detected tables to CSV  
3. **Vision‑Language Agent** – interprets molecular images via DashScope (Tongyi)  
4. **OCR Agent** – reads text from paper fragments  

The extracted data can then be converted to SMILES (using MolScribe) and used for downstream machine learning tasks.

---

## 🛠️ Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install easyocr==1.7.2 opencv-python==4.8.0.74 pdf2image==1.16.3 dashscope==1.25.2
```

> **Note:** `pdf2image` requires `poppler` to be installed on your system ([installation guide](https://pdf2image.readthedocs.io/en/latest/installation.html)).

---

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
- `[gpu_id]` – optional GPU ID for YOLO inference (default: CPU)

After execution, extracted table data (JSON) will be placed inside the `table_image` folder under your output directory.

---

## 🔬 Detailed Step‑by‑Step Extraction

If you prefer to run each agent separately, follow the steps below. All commands assume you are in the project root.

### Step 1: Preprocess PDFs (Figure/Table Detection)

```bash
python main_extract_oled_preprocess.py \
    --model_pt your_YOLO/weights/best.pt \
    --yolo_project_path your_YOLO_master \
    --pdf_dir /data/pdf_examples/ \
    --output_dir output/pdf_extract/run_test
```

This step uses a YOLO model to detect tables and molecular structure images in each page of the PDFs. Detected regions are saved as cropped images.

### Step 2: Extract Tables to CSV

```bash
python main_extract_pdf_csv_only.py \
    --dir2process output/pdf_extract/run_test \
    --pdf_dir /data/pdf_examples/ \
    --skip_n 0
```

Converts the detected table images into CSV files using OCR and post‑processing.

### Step 3: Interpret Molecular Images with DashScope (Tongyi)

```bash
python main_tongyi_extract_img2json.py \
    --dir2process output/pdf_extract/run_test \
    --skip_n 0
```

Sends cropped molecular images to the DashScope API (requires proper configuration of API keys) and receives structured descriptions in JSON format.

### Step 4: OCR Text Extraction

```bash
python main_extract_oled_ocr.py \
    --dir2process output/pdf_extract/run_test \
    --pdf_dir /data/pdf_examples/ \
    --skip_n 0 \
    --gpu 0
```

Performs OCR on the remaining text regions to capture captions, labels, and property values.

---

## 🧪 Molecular Structure → SMILES Conversion

After the four agents have run, you can convert the extracted molecular structure images into SMILES strings using **MolScribe**.

**Example command:**

```bash
python run_detect_cpu_for_pdf_dirs.py \
    --ckpt 'MolScribe/model/swin_base_char_aux_1m.pth' \
    --dir2process output/pdf_extract/run_test \
    --skip_n 0
```

- Extracted molecule images are saved under `<output_dir>/molecule_images` (configurable).  
- The resulting molecule‑property JSON files are placed in the `table_image` folder alongside the table data.

> 💡 Make sure you have the MolScribe repository and pre‑trained weights downloaded. See [MolScribe](https://github.com/thomas0809/MolScribe) for details.

---

## 🤖 Downstream Machine Learning

The `MLs/` folder contains **six curated OLED material property datasets** (TADF molecules) used in our paper. You can use these datasets directly or generate your own from extracted data.

To create a training dataset from your own extracted JSON files:

```bash
python MLs/create_dataset_mol_get_properties.py \
    --input_dir output/pdf_extract/run_test \
    --output_csv my_dataset.csv
```

This script parses the molecule‑property JSONs and compiles them into a clean CSV ready for ML models.

---

## 📁 Output Structure

A typical output directory (`output/pdf_extract/run_test`) looks like:

```
run_test/
├── table_image/           # JSON files for extracted tables
├── molecule_images/       # cropped molecular structure images
├── ocr_text/              # raw OCR outputs
├── csv_tables/            # CSV versions of tables
└── logs/                  # processing logs
```

---

## ✏️ Customization

Each agent is designed to be modular. You can:
- Replace the YOLO model with a custom detector.
- Swap the DashScope agent with another vision‑language model.
- Modify OCR parameters in `main_extract_oled_ocr.py` to suit different PDF qualities.

See the comments inside each script for detailed configuration options.

---

## 📚 Citation

If you use OLED-MAMA in your research, please cite our paper (to be added).

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*Happy extracting!* 🧪🔍




