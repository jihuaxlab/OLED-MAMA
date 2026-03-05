# preprocess -> preprocess tables -> vllm -> ocr -> mol recognization
# python main_extract_oled_ocr.py  --dir2process /mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/output/pdf_extract/run_test --pdf_dir /mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/data/pdf_examples/run_test --skip_n 0 --gpu 0
import cv2
import argparse
from datetime import datetime
import os
import re
import json
from tqdm import tqdm
import difflib
from collections import defaultdict
import tools
import shutil
from datetime import datetime
import easyocr

ocr_prob = 0.3

output_table_img_dir_name = 'table-images'
output_mol_img_dir_name = 'cropped_images'
final_mol_img_dir_name = 'extracted_images_fix'
final_mol_for_ai_dir_name = 'extracted_images_for_ai'
final_ocr_json = 'ocr_closest_to_center_fix.json'


def sanitize_filename(text, space_replacement='_'):
    # 移除非法字符
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
    # 压缩空白并替换空格
    cleaned = ' '.join(cleaned.split())
    if space_replacement:
        cleaned = cleaned.replace(' ', space_replacement)
    # 避免以 . 或 - 开头/结尾（可选）
    cleaned = cleaned.strip('-.')
    # 避免空文件名
    if not cleaned:
        cleaned = "unnamed"
    return cleaned


def normalize_for_matching(name: str) -> str:
    """
  将原始 PDF 文件名标准化，便于和 folder_name 匹配：
  - 移除 .pdf
  - 替换多个空格为单个空格
  - 移除特殊字符（保留字母、数字、空格、中文、-）
  """
    name = os.path.splitext(name)[0]  # 去掉 .pdf
    # 保留中文、英文、数字、空格、连字符，其他替换为空格
    name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def folder_to_query(folder_name: str) -> str:
    """
  将规范化文件夹名转回“可匹配”的查询字符串：
  - 将 _ 替换为空格
  - 但保留 "-_" -> "-"
  """
    # 先把 "-_" 和 "_-" 还原为 "-"
    s = folder_name.replace("_-_", " - ")
    s = s.replace("_-", " -")
    s = s.replace("-_", "- ")
    # 再把剩余的 _ 替换为空格
    s = s.replace("_", " ")
    # 清理多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_matching_pdf(folder_name, normalized_pdfs, top_n=3) -> list:
    """
  在 pdf_dir 中查找与 folder_name 最匹配的 PDF 文件
  """
    query = folder_to_query(folder_name)
    print(f"🔍 查询字符串: '{query}'")

    # 获取所有 PDF 文件名（标准化后）

    # 提取标准化后的名称用于匹配
    norm_names = [item[1] for item in normalized_pdfs]

    # 使用 difflib 模糊匹配
    matches = difflib.get_close_matches(query, norm_names, n=top_n, cutoff=0.6)

    # 返回原始文件名
    matched_files = []
    for match in matches:
        for orig_file, norm in normalized_pdfs:
            if norm == match:
                matched_files.append(orig_file)
                break

    return matched_files


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


def get_run_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir2process', type=str, default='',
                      help="ALL PDF directories after csv and figures extraction.")
  parser.add_argument('--skip_n', type=int, default=0,
                      help="Resume process.")
  parser.add_argument('--gpu', type=int, default=0,
                      help="Single GPU for Easyocr.")
  parser.add_argument('--pdf_dir', type=str, default='',
                      help="ALL PDF files.")
  parser.add_argument('--process_csv', action='store_true', help="LLM extract info to json.")
  opt = parser.parse_args()
  return opt

def main(opt):
  root_path = opt.dir2process
  all_pdfs_dir = os.listdir(root_path)
  skip_i = opt.skip_n

  all_source_pdf_files = [f for f in os.listdir(opt.pdf_dir) if f.lower().endswith('.pdf')]

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

    pdf_names = [f for f in os.listdir(output_dir) if f.lower().endswith('.pdf')]
    if len(pdf_names) > 0:
      pdf_name = pdf_names[0]
      print(f"当前文件夹{pdf_dir_name}匹配的pdf名称为：{pdf_name}")
      if pdf_name not in all_source_pdf_files:
        print(f"当前文件夹{pdf_dir_name}匹配的pdf名称失败，未找到原始pdf文件！")
        continue
    else:
      print(f"当前文件夹{pdf_dir_name}匹配的pdf名称失败，目标文件夹不包含pdf文件！")
      continue
    # pdf_path = os.path.join(opt.pdf_dir, pdf_name)

    output_mol_crop_dir = os.path.join(output_dir, output_mol_img_dir_name)
    if not os.path.isdir(output_mol_crop_dir):
      print(f"当前文件夹{output_mol_crop_dir}不存在！")
      continue

    output_tables_crop_dir = os.path.join(output_dir, output_table_img_dir_name)
    if not os.path.isdir(output_tables_crop_dir):
      print(f"当前文件夹{output_tables_crop_dir}不存在！")
      continue

    final_mol_dir = os.path.join(output_dir, final_mol_img_dir_name)
    if os.path.exists(final_mol_dir):
        shutil.rmtree(final_mol_dir)
    os.mkdir(final_mol_dir)

    final_mol_for_ai_dir = os.path.join(output_dir, final_mol_for_ai_dir_name)
    if os.path.exists(final_mol_for_ai_dir):
        shutil.rmtree(final_mol_for_ai_dir)
    os.mkdir(final_mol_for_ai_dir)

    output_temp_dir = os.path.join(output_dir, 'tmp')
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)

    all_mol_keys = set()
    json_files = [f for f in os.listdir(output_tables_crop_dir) if f.endswith('.json')]
    for jf in json_files:
        with open(os.path.join(output_tables_crop_dir, jf), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'All_Mol' in data and isinstance(data['All_Mol'], dict):
                    all_mol_keys.update(data['All_Mol'].keys())
            except Exception as e:
                print(f"⚠️ JSON 解析失败 {jf}: {e}")
                continue

    if not all_mol_keys:
        print(f"⚠️ 当前文件夹 {pdf_dir_name} 未找到任何分子名称 (All_Mol)")
        continue

    print(f"🔍 提取到 {len(all_mol_keys)} 个候选分子名称: {sorted(list(all_mol_keys))}")
    key_patterns = tools.build_core_patterns_from_examples(list(all_mol_keys))
    print('key_patterns:', key_patterns)

    # Step 2: 按页面分组分子图像
    mol_images = [f for f in os.listdir(output_mol_crop_dir) if f.endswith('.jpg')]
    page_groups = defaultdict(list)
    for img in mol_images:
        # 文件名格式: page_10_crop0_cls0.jpg
        match = re.search(r'page_(\d+)_crop\d+_cls\d+\.jpg', img)
        if match:
            page_id = int(match.group(1))
            page_groups[page_id].append(img)
        else:
            print(f"⚠️ 无法解析图像页码: {img}")
    if len(page_groups) == 0:
        print(f"⚠️ 无法解析图像页码")
        continue
    # print(page_groups)

    # Step 3: 对每张分子图像运行 OCR，并匹配候选名称
    # 存储所有有效 OCR 匹配结果
    filtered_valid_keys_dict = defaultdict(list)
    for page_id, img_list in page_groups.items():
        valid_keys_dict = defaultdict(list)  # 用来记录单页所有图的数据
        directions = {'bottom': 0, 'top': 0, 'right': 0, 'left': 0}
        for img_file in img_list:
            img_path = os.path.join(output_mol_crop_dir, img_file)
            try:
                image = cv2.imread(img_path)
                h, w = image.shape[:2]
                print("处理", img_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gray_img_path = os.path.join(output_temp_dir, f'gray_captcha_{timestamp}.png')
                cv2.imwrite(gray_img_path, gray)
                reader = easyocr.Reader(
                    ['en'],
                    model_storage_directory="/home/hezhaoming/.EasyOCR/model",
                    gpu=True)
                # 'ch_sim' 模型对 T vs J 区分能力弱 字符粘连或断裂	- 和 3 粘在一起，或 T 的横线缺失
                results = reader.readtext(gray_img_path, detail=1)

                if not results or not results[0]:
                    continue
                boxes_and_txts = []
                for (bbox, text, prob) in results:
                    if prob < ocr_prob:
                        continue
                    # print(f"文本: {text}")
                    # print(f"置信度: {prob:.4f}")
                    # print(f"位置框: {bbox}")  # bbox 是一个包含 4 个点的列表
                    if not text:
                        continue
                    # 转为 (x_min, y_min, x_max, y_max)
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    # top_left = np.array(bbox[0]) - np.array([w / 2, h / 2])
                    # bottom_right = np.array(bbox[2]) - np.array([w / 2, h / 2])
                    # d_ocr_result_2_center = np.linalg.norm((top_left + bottom_right) / 2)
                    # print(top_left, bottom_right)
                    boxes_and_txts.append((text, (x1, y1, x2, y2)))

                img_center = (w / 2, h / 2)
                # 提取所有文本框和内容
                min_distance = 999999
                closest_text_center = None
                closest_text = None
                # 对每个 OCR 文本，检查是否匹配 any candidate key (模糊匹配)
                for txt, bbox in boxes_and_txts:
                    # 清理 OCR 文本（分割空格）
                    clean_txt = txt.strip()
                    if len(clean_txt) == 0:
                        continue
                    # 尝试精确或模糊匹配
                    matched_key = False
                    if clean_txt in all_mol_keys:
                        matched_key = clean_txt
                    else:
                        # 尝试模糊匹配（用re查找所有可能的名称）
                        matched_key, matched_list = tools.is_candidate_valid(txt, key_patterns, max_extra=2)
                        if matched_key:
                            matched_key = ' '.join(matched_list)
                            # print("匹配--key：", matched_list, txt)
                        else:
                            pass
                            # print("不匹配--key：", clean_txt)

                    if matched_key:
                        x1, y1, x2, y2 = bbox
                        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        # 计算到图像中心的欧氏距离
                        dist = ((bbox_center[0] - img_center[0]) ** 2 + (
                                    bbox_center[1] - img_center[1]) ** 2) ** 0.5
                        valid_keys_dict[matched_key].append({
                            'bbox': [x1, y1, x2, y2],
                            'bbox_center': bbox_center,
                            'img_wh': [w, h],
                            'bbox_ref_distance': round(dist, 2),
                            'page_id': page_id,
                            'img_file_name': img_file
                        })
                        if dist < min_distance:
                            closest_text_center = bbox_center
                            closest_text = matched_key

                if closest_text_center is not None:
                    center_y = closest_text_center[1]
                    center_x = closest_text_center[0]
                    # 判断主要方位（取最显著的一个）
                    if center_y < h * 0.35:
                        directions['top'] += 1
                        print(f"closest_text {closest_text} up")
                    if center_y > h * 0.65:
                        directions['bottom'] += 1
                        print(f"closest_text {closest_text} bottom")
                    if center_x < w * 0.35:
                        directions['left'] += 1
                        print(f"closest_text {closest_text} left")
                    if center_x > w * 0.65:
                        directions['right'] += 1
                        print(f"closest_text {closest_text} right")
                else:
                    pass
                    # print("closest_text_center not Found")

            except Exception as e:
                print(f"⚠️ OCR 失败 {img_path}: {e}")
                continue
        # 确定全局主导方向
        dominant_dir = max(directions, key=directions.get)
        print(f"✅ 全局主导图注方向: {dominant_dir} (scores: {directions})")

        # Step 5: 过滤掉不符合主导方向的匹配项
        for key, matches in valid_keys_dict.items():
            for m in matches:
                w, h = m['img_wh']
                center_x, center_y = m['bbox_center']
                # print(key, m)
                keep = False
                if dominant_dir == 'top' and center_y < h * 0.4:
                    keep = True
                elif dominant_dir == 'bottom' and center_y > h * 0.6:
                    keep = True
                elif dominant_dir == 'left' and center_x < w * 0.4:
                    keep = True
                elif dominant_dir == 'right' and center_x > w * 0.6:
                    keep = True
                if keep:
                    filtered_valid_keys_dict[key].append(m)

        # # Step 6: 检查如果有些图没有主导方向的key则可以主导方向顺序进行保留最近的一个key
        # retained_img_files = set()
        # for key, matches in filtered_valid_keys_dict.items():
        #     for m in matches:
        #         if m['page_id'] == page_id:
        #             retained_img_files.add(m['img_file_name'])
        #
        # # 遍历本页所有图像
        # for img_file in img_list:
        #     if img_file in retained_img_files:
        #         continue  # 已有保留项，跳过
        #
        #     # 收集该图所有原始匹配项
        #     candidates = []
        #     for key, matches in valid_keys_dict.items():
        #         for m in matches:
        #             if m['img_file_name'] == img_file:
        #                 candidates.append((key, m))
        #
        #     if not candidates:
        #         continue  # 该图本来就没有匹配项
        #
        #     # 选距离图像中心最近的一个
        #     best_key, best_match = min(candidates, key=lambda item: item[1]['bbox_ref_distance'])
        #     filtered_valid_keys_dict[best_key].append(best_match)
        #     print(f"⚠️ Step6 fallback (nearest): {img_file} -> {best_key}")

        # print(filtered_valid_keys_dict)

    # 读取output_tables_crop_dir所有json文件，收集"All_Mol":{XXX}的keys 放入 all keys = []
    # 读取 output_mol_crop_dir 所有 jpg 按照page分组page_页面编号xxx_cropxxxxxx.jpg 文件
    # 先为每个图像过一遍OCR，按照相似度判断OCR结果是否在all keys中，并将每个图中符合条件的ocr结果和所在的bbox，以及bbox的中心点到图像中心点相对位置，所在页码进行记录到valid_keys_dict中
    # valid_keys_dict = {‘ocr文本结果’:[{bbox: [], bbox_ref_distance: xx（bbox中心距离图像中心的距离）, page_id: xxx, img_file_name: 'xxx'}, 可能有多个图中都出现了,或者一个图出现多次], ...}
    # 下面按照每个页面判个keys出现在每个页面的分子图像的方位
    # for page_i in all_page:
    # 计算keys在 下上右左 方位的可行度
    # p_list = cal_proba(page_i, 其他参数)
    # 寻找p_list 最大的放位进行判定每个图中ocr结果是否能够和本图像对应，例如本页的分子图注都应该在图像的下方，则bbox 的y大于图像 1/2的文本都无效
    # 如果图中不包含有效的keys则跳过
    # 将图中该假设方向的成立的进行保留‘ocr文本结果’:[{bbox: [], bbox_ref_xy: [], page_id: xxx, img_file_name: 保留符合条件的图像路径}]
    # 检查如果有图没有key则取最近的一个key作为该图的描述

    # cal_proba(page_i, valid_keys_dict， 其他可能的参数) 如何写
    # 将本页的所有图进行四个方位的假设，例如假如是下方，则统计本页所有图像中处于valid_keys_dict中文本处于图像下方且距离图像中心点最近的比例，也就是每个图下方存在一个
    # 距离图像中心最近的key。（先前已经计算了bbox_ref_distance，只需要比较一个图中各个key的bbox_ref_distance，如果最小的key的y坐标在图像在图像下方就说明判断正确+1否则为错误+1）
    # 其他方向假设也是相同道理，统计每个图符合该假设的比例作为 cal_proba的四个输出比例。

    if len(filtered_valid_keys_dict) == 0:
        print(f"⚠️ 未在任何分子图像中通过 OCR 找到匹配的分子名称")

    else:
        # 基于filtered_valid_keys_dict记录的情况将key对应的图像进行合并，比如'C4Hg': [{'bbox': [500, 306, 552, 330], 'bbox_center': (526.0, 318.0), 'img_wh': [556, 386], 'bbox_ref_distance': 277.72, 'page_id': 10, 'img_file_name': 'page_10_crop12_cls0.jpg'}, {'bbox': [576, 418, 628, 444], 'bbox_center': (602.0, 431.0), 'img_wh': [732, 508], 'bbox_ref_distance': 295.0, 'page_id': 10, 'img_file_name': 'page_10_crop15_cls0.jpg'}
        # 需要把对应的图（如果有多个保留前三个）按distanc次序进行纵向拼接，然后保存到final_mol_dir路径下
        final_mol_dir = os.path.join(output_dir, final_mol_img_dir_name)
        if not os.path.isdir(final_mol_dir):
            os.mkdir(final_mol_dir)
        # 然后把结果写入path
        final_json_path = os.path.join(output_dir, final_ocr_json)
        # 格式如下
        # {
        #     "S-CNDF-S-tCz": {
        #         "distance": 284取整,
        #         "image_path": [f"{final_mol_img_dir_name}/keys.jpg", ...],
        #         "image_info"：["page_2_crop1_cls0.jpg", ...]
        #     },
        # }
        results = {}
        for key, matches in filtered_valid_keys_dict.items():
            # 按bbox_ref_distance排序并最多保留前2个
            sorted_matches = sorted(matches, key=lambda x: x['bbox_ref_distance'])[:2]

            # 如果没有匹配项或所有匹配项都被过滤掉，则跳过该key
            if not sorted_matches:
                continue

            # 初始化用于纵向拼接的图像列表
            images_to_merge = []

            # 记录每个key的图像路径和距离信息
            image_paths = []
            image_infos = []
            distance_infos = []

            loaded_img = []
            for match in sorted_matches:
                img_path = os.path.join(output_mol_crop_dir, match['img_file_name'])
                if img_path in loaded_img:
                    continue
                image = cv2.imread(img_path)
                loaded_img.append(img_path)
                if image is None:
                    print(f"警告：无法读取图像 {img_path}")
                    continue
                images_to_merge.append(image)
                image_paths.append(f"{final_mol_img_dir_name}/keys.jpg")  # 合并后的图像路径
                image_infos.append(match['img_file_name'])  # 原始图像文件名
                distance_infos.append(match['bbox_ref_distance'])

            # 纵向拼接图像
            if images_to_merge:
                max_width = max([img.shape[1] for img in images_to_merge])
                padded_images = []
                for img in images_to_merge:
                    h, w = img.shape[:2]
                    pad_left = (max_width - w) // 2
                    pad_right = max_width - w - pad_left  # ✅ 精确保证总宽 = max_width
                    padded = cv2.copyMakeBorder(
                        img,
                        top=0, bottom=5,
                        left=pad_left, right=pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=(0, 0, 255)
                    )
                    padded_images.append(padded)

                merged_image = cv2.vconcat(padded_images)

                # 保存合并后的图像
                merged_image_key = sanitize_filename(key)
                merged_image_name = f"{merged_image_key}.jpg"
                merged_image_path = os.path.join(final_mol_dir, merged_image_name)
                cv2.imwrite(merged_image_path, merged_image)

                # === 拷贝最近的一个分子图像到final_mol_for_ai_dir ===
                closest_match = sorted_matches[0]
                closest_img_path = os.path.join(output_mol_crop_dir, closest_match['img_file_name'])
                ai_img_dest_path = os.path.join(final_mol_for_ai_dir, closest_match['img_file_name'])

                try:
                    shutil.copy(closest_img_path, ai_img_dest_path)
                except Exception as e:
                    print(f"❌ 拷贝最近的分子图像时出错: {e}")

                # 将结果写入results字典
                results[key] = {
                    "distance": distance_infos,
                    "image_path": [f"{final_mol_img_dir_name}/{merged_image_name}"],  # 合并后的图像路径
                    "image_info": image_infos,  # 原始图像文件名
                }

        # 写入JSON文件
        final_json_path = os.path.join(output_dir, final_ocr_json)
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"✅ 结果已保存到 {final_json_path}")



if __name__ == "__main__":
  args = get_run_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  #  easyocr 只使用 GPU 0
  main(opt=args)



