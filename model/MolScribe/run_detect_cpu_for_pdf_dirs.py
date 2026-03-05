# 使用外部程序调用,用于处理批量提取pdf信息后的分子结构识别
# python run_detect_cpu_for_pdf_dirs.py --ckpt '/mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/mol_recognize/MolScribe/model/swin_base_char_aux_1m.pth' --dir2process '/mnt/LargeStorageSpace/HEZhaoming/decode_chem_pdf/data/all-TADF-ref/outputs/' --skip_n 0
from molscribe import MolScribe  # import 先后顺序很重要 需要先import这个库 再import torch
import torch
import argparse
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
import os
import shutil

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from molscribe.dataset import get_transforms

import json
from tqdm import tqdm

import re
from datetime import datetime

pdf_extracted_images_dir = 'extracted_images'  # 文件夹名称由批量提取pdf信息后的分子结构识别脚本中定义
pdf_extracted_images_dir = 'extracted_images_for_ai'  # 文件夹名称由批量提取pdf信息后的分子结构识别脚本中定义

with_denoise = False
with_prepocess = True
ocr_expand_rate = 1.5  # 截取分子时候扩大范围用于保存分子名称标注，在识别分子结构时候需要裁剪多余部分
white_threshold = 235 # 大于该值视为背景

if with_denoise:
    print("使用去噪处理图像")
else:
    print("不使用去噪处理图像")

if with_prepocess:
    print("使用去白边预处理图像")
else:
    print("不使用去白边预处理图像")


def get_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./model/swin_base_char_aux_1m.pth')
    parser.add_argument('--dir2process', type=str, default='',
                        help="ALL PDF directories after csv and figures extraction.")
    parser.add_argument('--skip_n', type=int, default=0,
                        help="Resume process.")
    opt = parser.parse_args()
    return opt


def get_args(args_states=None):
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--encoder', type=str, default='swin_base')
    parser.add_argument('--decoder', type=str, default='transformer')
    parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--enc_pos_emb', action='store_true')
    group = parser.add_argument_group("transformer_options")
    group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
    group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
    group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
    group.add_argument("--dec_num_queries", type=int, default=128)
    group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
    group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
    group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
    parser.add_argument('--continuous_coords', action='store_true')
    parser.add_argument('--compute_confidence', action='store_true')
    # Data
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--coord_bins', type=int, default=64)
    parser.add_argument('--sep_xy', action='store_true', default=True)

    args = parser.parse_args([])
    if args_states:
        for key, value in args_states.items():
            args.__dict__[key] = value
    return args


def is_light_color(pixel, low_threshold=200):
    """
    判断像素是否为浅色。
    :param pixel: 单个像素值或像素数组，假设是RGB格式。
    :param low_threshold: 浅色判断阈值，默认200。
    :return: 布尔值或布尔数组，表示像素是否为浅色。
    """
    return np.all(pixel >= low_threshold, axis=-1)


def find_edge(image, direction, low_threshold=200):
    """
    从中心开始向指定方向寻找边缘。
    :param image: 输入图像，numpy数组形式，形状为(H, W, C)。
    :param direction: 寻找方向，可以是'up', 'down', 'left', 'right'。
    :param low_threshold: 浅色判断阈值，默认200。
    :return: 边缘位置的索引。
    """
    check_percentage = 0.03
    h, w, c = image.shape
    h_check = max(3, int(h * check_percentage)) # 检查连续浅色的行数
    w_check = max(3, int(w * check_percentage))
    # print("h_check, w_check")
    # print(h_check, w_check, h, w)
    mid_h, mid_w = h // 2, w // 2

    if direction == 'up':
        for i in range(mid_h - 1, -1, -1):
            if np.all(is_light_color(image[i, :, :], low_threshold)):
                # print('白线', i)
                is_edge = True
                if i == 0:
                    return 0
                for j in range(max(0, i - h_check), i):
                    # print('校验up白线', j)
                    if not np.all(is_light_color(image[j, :, :], low_threshold)):
                        # print('非白线', j)
                        is_edge = False
                        break
                if is_edge:
                    return i
        return 0
    elif direction == 'down':
        for i in range(mid_h, h):
            if np.all(is_light_color(image[i, :, :], low_threshold)):
                is_edge = True
                if i == h-1:
                    return h
                for j in range(i, min(h-1, i + h_check)):
                    if not np.all(is_light_color(image[j, :, :], low_threshold)):
                        is_edge = False
                        break
                if is_edge:
                    return i+1
        return h
    elif direction == 'left':
        for i in range(mid_w - 1, -1, -1):
            if np.all(is_light_color(image[:, i, :], low_threshold)):
                is_edge = True
                if i == 0:
                    return i
                for j in range(max(0, i - w_check), i):
                    if not np.all(is_light_color(image[:, j, :], low_threshold)):
                        is_edge = False
                        break
                if is_edge:
                    return i
        return 0
    elif direction == 'right':
        for i in range(mid_w, w):
            if np.all(is_light_color(image[:, i, :], low_threshold)):
                is_edge = True
                if i == w - 1:
                    return h
                for j in range(i, min(w - 1, i + w_check)):
                    if not np.all(is_light_color(image[:, j, :], low_threshold)):
                        is_edge = False
                        break
                if is_edge:
                    return i + 1
        return w


def remove_white_borders(image, pad_size=5, low_threshold=200):
    """
    移除图像的白边，并在周围填充指定大小的白色区域。
    :param image: 输入图像，numpy数组形式，形状为(H, W, C)。
    :param pad_size: 四周填充的大小，默认5。
    :param low_threshold: 浅色判断阈值，默认200。
    :return: 处理后的图像。
    """
    top = find_edge(image, 'up', low_threshold)
    bottom = find_edge(image, 'down', low_threshold)
    left = find_edge(image, 'left', low_threshold)
    right = find_edge(image, 'right', low_threshold)

    cropped_image = image[top:bottom, left:right]
    #
    # 使用白色填充
    # white_pixel = np.array([255, 255, 255])
    # padded_image = np.pad(cropped_image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=white_pixel)

    return cropped_image


def denormalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    image_tensor: (C, H, W) or (H, W, C) tensor or numpy array in normalized space
    Returns: (H, W, C) numpy array in [0, 1] range
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.cpu().numpy()
    else:
        image = image_tensor.copy()

    # 如果是 (C, H, W)，转为 (H, W, C)
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    # 反归一化: x = x * std + mean
    image = image * np.array(std) + np.array(mean)

    # 限制到 [0, 1] 防止溢出
    image = np.clip(image, 0, 1)
    return image


def convert_tensor_items_to_python_types(data):
    """
    递归地遍历字典或列表中的元素，并将其中的 Tensor 转换为 Python 数据类型。
    """
    if isinstance(data, dict):
        return {key: convert_tensor_items_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensor_items_to_python_types(element) for element in data]
    elif isinstance(data, torch.Tensor):
        # 如果 Tensor 是标量，使用 item() 方法转换为 Python 数值类型
        if data.dim() == 0:
            return data.item()
        # 否则，将 Tensor 转换为 Python 列表
        return data.tolist()
    else:
        return data


def enhance_image_back(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 转换为灰度图，以便于处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用非局部均值去噪
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    #    # 应用CLAHE来增强对比度
    #    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #    enhanced_contrast = clahe.apply(denoised)

    #    # 锐化图像
    #    kernel = np.array([[-1,-1,-1],
    #                       [-1, 9,-1],
    #                       [-1,-1,-1]])
    #    sharpened = cv2.filter2D(denoised, -1, kernel)

    # 保存增强后的图像
    cv2.imwrite(output_path, denoised)


def enhance_image(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 转换为灰度图，以便于处理
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoised = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)
    # We assume atoms are represented by darker pixels (black dots)
    # _, thresholded_image = cv2.threshold(denoised, 50, 255, cv2.THRESH_BINARY_INV) # 反色
    # _, out_image = cv2.threshold(denoised, 180, 255, cv2.THRESH_BINARY)
    # 腐蚀一下
    kernel = np.ones((1, 1), np.uint8)
    out_image = cv2.erode(gray_image, kernel, iterations=1)
    denoised = cv2.fastNlMeansDenoising(out_image, None, 10, 7, 21)
    # processed_image = np.where(denoised < 50, 0, denoised)

    cv2.imwrite(output_path, denoised)


def draw_molecule(transformed_image, atoms, bonds, save_path, input_size, json_path=None, smiles=None):
    img_vis = denormalize(transformed_image)
    # transformed_image = image
    # print(transformed_image.shape)  # 此处确定为 (3, 384, 384)

    if transformed_image.shape[0] == 3 and len(transformed_image.shape) == 3:
        # 转换为 (384, 384, 3)
        transformed_image = transformed_image.permute(1, 2, 0)

    img_width = input_size
    img_height = input_size

    # 将PIL图像转换为适合matplotlib使用的数组格式
    fig, ax = plt.subplots()
    ax.imshow(img_vis)
    ax.axis('off')  # 关闭坐标轴

    # 设置字体以便于显示文字
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # 根据实际情况调整字体大小
    except IOError:
        font = ImageFont.load_default()

    # 定义键类型的色彩
    bond_colors = {'single': 'blue', 'double': 'green'}  # 添加其他键类型的颜色

    # 绘制原子符号
    for atom in atoms:
        # print(atom['x']) # => 归一化坐标
        x_pixel = atom['x'] * img_width  # 将归一化x坐标转换为像素坐标
        y_pixel = atom['y'] * img_height  # 将归一化y坐标转换为像素坐标
        ax.text(x_pixel, y_pixel, atom['atom_symbol'],
                color='red', fontsize=6, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="white", edgecolor="black", alpha=0.5))

    # 绘制化学键
    for bond in bonds:
        start_atom = atoms[bond['endpoint_atoms'][0]]
        end_atom = atoms[bond['endpoint_atoms'][1]]
        ax.plot([start_atom['x'] * img_width, end_atom['x'] * img_width],
                [start_atom['y'] * img_height, end_atom['y'] * img_height],
                color=bond_colors.get(bond['bond_type'], 'red'), linewidth=2)  # 默认红色用于未定义的键类型
        confidence_text_x = (start_atom['x'] + end_atom['x']) / 2 * img_width
        conficence_text_y = (start_atom['y'] + end_atom['y']) / 2 * img_height
        ax.text(confidence_text_x, conficence_text_y, '{:.3f}'.format(bond['confidence']), color='red', fontsize=6)

    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    # print('保存预测详情到->', save_path)
    plt.close()  # 关闭图像以释放内存

    if json_path:
        output_dict = {}
        output_dict.update({'atom_info': atoms, 'bond_info': bonds})
        output_dict.update({'smiles': smiles})
        json_dict_convert = convert_tensor_items_to_python_types(output_dict)
        with open(json_path, 'w') as f:
            json.dump(json_dict_convert, f)
            f.close()


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


def main(opt):
    ckpt_path = opt.ckpt
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n
    print('加载识别模型...')
    model = MolScribe(ckpt_path, device=torch.device('cpu'))
    model_states = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = get_args(model_states['args'])
    # print('模型训练参数：', args)
    model_input_size = args.input_size
    # print( model_input_size) # => 384
    print('模型加载完毕.')

    print(f"处理文献文件夹共计数量{len(all_pdfs_dir)}")
    print(f"启动文件序号：{skip_i}")
    ii = 0
    for pdf_dir_name in tqdm(all_pdfs_dir):
        if ii < skip_i:
            ii += 1
            continue

        obj_dir = os.path.join(root_path, pdf_dir_name)
        detect_outputs = find_latest_molecules_result_folder(obj_dir)
        if detect_outputs is None:
            print("未找到目标检测结果文件夹!")
            continue
        figure_dir = os.path.join(root_path, pdf_dir_name, detect_outputs, pdf_extracted_images_dir)
        if not os.path.isdir(figure_dir):
            print(f"未找到分子文件夹{figure_dir},跳过!")
            continue
        output_dir = os.path.join(root_path, pdf_dir_name, detect_outputs, "mol_recognize_results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_log_dir = os.path.join(output_dir, f'mol-recognize-log-{timestamp}')
        os.makedirs(output_log_dir, exist_ok=True)
        result_json_path = os.path.join(output_dir, f'mol_recognize_result_{timestamp}.json')

        all_mol_figures = os.listdir(figure_dir)

        # image_transform = get_transforms(input_size, augment=False)
        recognize_result_dict = {}
        for im_name in all_mol_figures:
            # if im_name != 'page_8_crop4_cls0.jpg':
            #     continue
            original_image_path = os.path.join(figure_dir, im_name)
            base_name = os.path.basename(original_image_path)
            shutil.copy(original_image_path, output_log_dir)
            if with_denoise:
                enhance_output_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-enhanced.png")
                enhance_image(original_image_path, enhance_output_path)
                image_path = enhance_output_path
            else:
                image_path = original_image_path
            # 加载图像
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if with_prepocess:
                h_raw, w_raw = image.shape[:2]

                # 计算目标尺寸（整数）
                new_h = int(h_raw / ocr_expand_rate)
                new_w = int(w_raw / ocr_expand_rate)

                # 确保不超出原图范围
                new_h = min(new_h, h_raw)
                new_w = min(new_w, w_raw)

                # 计算中心裁剪的起始坐标
                start_y = (h_raw - new_h) // 2
                start_x = (w_raw - new_w) // 2

                # 裁剪
                cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
                image = remove_white_borders(cropped, low_threshold=white_threshold)  # 白边保持和训练中相同宽度 pad = 5

                # model.predict_image 中 image_transform(image=image, keypoints=[])['image'] # 主要有一个去白边 -> resize -> normalization 的操作参考 dataset.py
                # predict_image 的输出改了中间加了image_transform 后的图像
            pred = model.predict_image(image, return_atoms_bonds=True, return_confidence=True)
            output = pred[0]
            images = pred[-1][0][0]
            # print(images.shape)

            smiles = output['smiles']
            all_mol_detected = smiles.split('.')
            longest_smiles = max(all_mol_detected, key=len)

            # print("所有片段:", all_mol_detected)
            # print("最长的 SMILES:", longest_smiles)
            # print("长度:", len(longest_smiles))

            save_image_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-raw-prediction.png")
            save_json_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-atoms-bonds.json")
            draw_molecule(images, output['atoms'], output['bonds'], save_image_path, model_input_size, json_path=save_json_path,
                          smiles=longest_smiles)

            if smiles:
                mol = Chem.MolFromSmiles(longest_smiles)
            else:
                pass
                # print('识别失败')

            if mol is None:
                # pass
                print("❌ RDKit 解析 失败, 保存SMILES")
                # 绘图并保存为 PNG
                recognize_result_dict.update({im_name: {'smiles': longest_smiles,
                                                        'mol_fig_path': ""}})

            else:

                # 可选：生成更美观的坐标（力场优化）
                from rdkit.Chem import AllChem

                AllChem.Compute2DCoords(mol)

                # 绘图并保存为 PNG
                img = Draw.MolToImage(mol, size=(400, 400), kekulize=True)
                output_img_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-recognition.png")
                img.save(output_img_path)
                relative_image_path = os.path.relpath(output_img_path, output_dir)  # 使用相对路径保存
                recognize_result_dict.update({im_name: {'smiles': longest_smiles,
                                                        'mol_fig_path': relative_image_path}})
                print(f"✅ 图像{im_name}分子识别成功： {longest_smiles}")

        with open(result_json_path, 'w') as f:
            json.dump(recognize_result_dict, f, indent=4)
            f.close()
            # print(f'结果写入{result_json_path}')
            # raise


if __name__ == '__main__':
    run_opt = get_run_args()
    main(run_opt)

