# OLED-MAMA
This a multi-agent framework specifically designed for OLED material data extraction.

快速start
直接运行 extract.sh

提取的表格数据json会被放置再table_image文件夹

本工作具体按照下面顺序使用多个agents一步一步提取PDF文件夹中的分子数据
Step1 python main_extract_oled_preprocess.py
Step2 python main_extract_pdf_csv_only.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0
Step3 python main_tongyi_extract_img2json.py  --dir2process your_output_dir --skip_n 0 
Step4 python main_extract_oled_ocr.py  --dir2process your_output_dir --pdf_dir your_raw_pdf_dir --skip_n 0 --gpu 0

基于上述步骤的结果使用Molecule structure 进一步转换smiles 码

你可以根据自己的需求对agent的实际工作进行修改从而满足你的需求。

