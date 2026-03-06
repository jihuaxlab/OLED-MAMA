# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
import subprocess
import os
import sys
import yolo_plus_easyOCR_full_v4 as cv_lib
from collections import defaultdict
import cv2
import numpy as np
import easyocr
from tqdm import tqdm
import json
import shutil
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract molecular structures and tables from PDFs using YOLOv5 and OCR.")
    
    # Model and YOLO settings
    parser.add_argument("--model_pt", type=str, required=True, help="Path to the trained YOLOv5 .pt model file.")
    parser.add_argument("--yolo_project_path", type=str, required=True, help="Path to the YOLOv5 project directory.")
    
    # Input/Output directories
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing input PDF files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for extracted results.")
    
    # OCR and cropping parameters
    parser.add_argument("--ocr_prob", type=float, default=0.30, help="Confidence threshold for OCR text filtering.")
    parser.add_argument("--ocr_expand_rate", type=float, default=1.5, 
                        help="Expansion ratio for cropping regions around detected molecules (to include nearby labels).")
    
    # Table bounding box expansion
    parser.add_argument("--table_extend_rate_h", type=float, default=0.3, 
                        help="Vertical expansion margin (normalized) for table bounding boxes.")
    parser.add_argument("--table_extend_rate_w", type=float, default=0.15, 
                        help="Horizontal expansion margin (normalized) for table bounding boxes.")
    
    # Processing control
    parser.add_argument("--resume_from", type=int, default=0, 
                        help="Skip the first N PDF files (for resuming interrupted runs).")
    parser.add_argument("--dpi", type=int, default=200, 
                        help="Resolution (DPI) for PDF-to-image conversion.")
    parser.add_argument("--cuda_device", type=str, default="0", 
                        help="GPU device ID(s) to use (e.g., '0' or '0,1').")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # Validate input PDF directory
    if not os.path.isdir(args.pdf_dir):
        print(f"Error: PDF directory does not exist: {args.pdf_dir}")
        sys.exit(1)

    all_files = [f for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')]
    file_count = len(all_files)
    start_index = args.resume_from

    if start_index > 0:
        print(f"Resuming from file index: {start_index}")

    for idx, pdf_file_name in enumerate(all_files):
        if idx < start_index:
            continue

        print("-" * 40)
        print(f"Processing file {idx + 1} / {file_count}: {pdf_file_name}")
        print("-" * 40)

        pdf_path = os.path.join(args.pdf_dir, pdf_file_name)

        # Convert PDF to images
        _, pdf_images_dir = cv_lib.pdf_to_images(
            pdf_path=pdf_path,
            output_folder=args.output_dir,
            dpi=args.dpi,
            first_page=None,
            last_page=None,
            output_name_max_len=50
        )

        if pdf_images_dir is None:
            print(f"Skipping file due to conversion failure: {pdf_path}")
            continue

        # Create timestamped output subdirectory for detection results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        molecules_detect_results_dir = os.path.join(pdf_images_dir, f"molecules_detect_results_{timestamp}")
        
        # Run YOLO detection and cropping (with molecule label expansion)
        cv_lib.run_detect_and_crop_v5(
            image_folder=pdf_images_dir,
            pdf_path=pdf_path,  # Required for CSV extraction in newer version
            output_folder=molecules_detect_results_dir,
            weights=args.model_pt,
            yolo_project_dir=args.yolo_project_path,
            expand_ratio=args.ocr_expand_rate
        )

        # Prepare directories for final extracted images
        cropped_images_dir = os.path.join(molecules_detect_results_dir, "cropped_images")
        extracted_image_dir = os.path.join(molecules_detect_results_dir, "extracted_images")
        os.makedirs(extracted_image_dir, exist_ok=True)

        # List valid cropped image files
        cropped_img_extensions = ['.jpg']
        all_images = [
            img for img in os.listdir(cropped_images_dir)
            if os.path.isfile(os.path.join(cropped_images_dir, img))
            and os.path.splitext(img)[1].lower() in cropped_img_extensions
        ]

        # TODO: Integrate LLM-based table data extraction and probability matching here
        # (Current placeholder for future implementation)

    print("All PDFs processed.")


if __name__ == "__main__":
    main()


