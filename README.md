# OLED-MAMA
This a multi-agent framework specifically designed for OLED material data extraction.

按照下面顺序一步一步提取PDF文件夹中的分子数据
Step1 python main_extract_oled_preprocess.py
Step2 python main_extract_pdf_csv_only.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0
Step3 python main_tongyi_extract_img2json.py  --dir2process your_output_dir --skip_n 0 
Step4 python main_extract_oled_ocr.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0 --gpu 0
Step5 
