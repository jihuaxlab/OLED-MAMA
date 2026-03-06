#!/bin/bash

# ./run_pipeline.sh <raw_pdf_dir> <yolo_weight> <yolo_project> [output_dir] [gpu_id]
# Example: ./run_pipeline.sh /data/pdf_examples/ yolov8s/best.pt /opt/YOLOv8 output/pdf_extract/run_test 0

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <raw_pdf_dir> <yolo_weight_path> <yolo_project_path> [output_dir] [gpu_id]"
    echo "Example: $0 /data/pdf_examples/ weights/best.pt /home/user/YOLOv8 output/pdf_extract/run_test 0"
    exit 1
fi

PDF_DIR="$1"
YOLO_WEIGHT="$2"
YOLO_PROJECT="$3"
OUTPUT_DIR="${4:-output/pdf_extract/run_test}"  # default
GPU_ID="${5:-0}"                                # default GPU 0

echo "📁 Raw PDF directory: $PDF_DIR"
echo "📦 YOLO weight: $YOLO_WEIGHT"
echo "📂 YOLO project path: $YOLO_PROJECT"
echo "📤 Output directory: $OUTPUT_DIR"
echo "🖥️  GPU ID: $GPU_ID"
echo ""

# Step 1
echo "🚀 Step 1: Preprocessing PDFs with YOLO..."
python main_extract_oled_preprocess.py \
  --model_pt "$YOLO_WEIGHT" \
  --yolo_project_path "$YOLO_PROJECT" \
  --pdf_dir "$PDF_DIR" \
  --output_dir "$OUTPUT_DIR"

# Step 2
echo "✅ Step 1 done. Starting Step 2..."
python main_extract_pdf_csv_only.py \
  --dir2process "$OUTPUT_DIR" \
  --pdf_dir "$PDF_DIR" \
  --skip_n 0

# Step 3
echo "✅ Step 2 done. Starting Step 3..."
python main_tongyi_extract_img2json.py \
  --dir2process "$OUTPUT_DIR" \
  --skip_n 0

# Step 4
echo "✅ Step 3 done. Starting Step 4..."
python main_extract_oled_ocr.py \
  --dir2process "$OUTPUT_DIR" \
  --pdf_dir "$PDF_DIR" \
  --skip_n 0 \
  --gpu "$GPU_ID"

echo "🎉 Pipeline finished successfully!"
