# OLED-MAMA
This a multi-agent framework specifically designed for OLED material data extraction.

项目主要使用的环境
easyocr                   1.7.2
opencv-python             4.8.0.74
pdf2image                 1.16.3
dashscope                 1.25.2

快速start
直接运行 extract.sh

提取的表格数据json会被放置在table_image文件夹

本工作具体按照下面顺序使用多个agents一步一步提取PDF文件夹中的分子数据
Step1 python main_extract_oled_preprocess.py --model_pt your_YOLO/weights/best.pt --yolo_project_path your_YOLO_master  --pdf_dir /data/pdf_examples/ --output_dir output/pdf_extract/run_tes
Step2 python main_extract_pdf_csv_only.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0
Step3 python main_tongyi_extract_img2json.py  --dir2process your_output_dir --skip_n 0 
Step4 python main_extract_oled_ocr.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0 --gpu 0

基于上述步骤的结果使用Molecule structure 进一步转换smiles 码。提取的分子名称-匹配图像结果会放置在xxx，提取的分子名称-属性Json结果会放置在table_image文件夹

你可以根据自己的需求对agent的实际工作进行修改从而满足你的需求。


