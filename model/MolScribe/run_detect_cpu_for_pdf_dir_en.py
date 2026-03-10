# python run_detect_cpu_for_pdf_dirs_en.py --ckpt 'MolScribe/model/swin_base_char_aux_1m.pth' --dir2process your_output_dir --skip_n 0
from molscribe import MolScribe  # Importing modules, the order is important. Import molscribe before torch.
import torch
import argparse
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
import os
import shutil
import subprocess
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from molscribe.dataset import get_transforms

import json
from tqdm import tqdm

import re
from datetime import datetime

pdf_extracted_images_dir = 'extracted_images_for_ai'  # Folder name defined in script for batch extraction of PDF information.

with_denoise = False # Image preprocessing not recommended
with_prepocess = True
ocr_expand_rate = 1.5  # Enlarge the scope when cropping molecules to preserve labels, unnecessary parts need to be trimmed during structure recognition.
white_threshold = 235 # Values greater than this are considered background.

# verified script
mol_recognize_postprocess = 'ring_validation.py'

if with_denoise:
    print("Image denoising processing will be used.")
else:
    print("No image denoising processing.")

if with_prepocess:
    print("Preprocessing images by removing white borders will be used.")
else:
    print("Not using preprocessing to remove white borders.")

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


def run_mol_postprocess_and_get_smiles(json_path, output_dir=None):
    """
    Call the mol_recognize_postprocess.py script, and then read the generated -SMILES.json file.
    Return (smiles_str, is_valid_bool)
    """
    cmd = ["python", mol_recognize_postprocess, '--json_path', json_path]
    if output_dir:
        cmd.extend(["--output_dir", output_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Child process failed:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return "", False

    # generated JSON path
    base_name = os.path.basename(json_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    smiles_json_path = os.path.join(output_dir, f"{base_name_no_ext}-SMILES.json")

    # read JSON
    if not os.path.exists(smiles_json_path):
        print(f"No SMILES result file was found.: {smiles_json_path}")
        return "", False

    with open(smiles_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    smiles = data.get("smiles", "")
    is_valid = data.get("is_valid", False)
    corr_img_path = data.get("img_path", "")
    print("is_valid: ", is_valid)
    return smiles, corr_img_path

def is_light_color(pixel, low_threshold=200):
    """
    Determine if a pixel or array of pixels is light.
    :param pixel: Single pixel value or array of pixels, assumed to be RGB format.
    :param low_threshold: Threshold for determining light color, default 200.
    :return: Boolean or boolean array indicating whether the pixel is light.
    """
    return np.all(pixel >= low_threshold, axis=-1)


def find_edge(image, direction, low_threshold=200):
    """
    Start from the center and search for the edge in the specified direction.
    :param image: Input image, in numpy array form, with shape (H, W, C).
    :param direction: Search direction, can be 'up', 'down', 'left', 'right'.
    :param low_threshold: Threshold for light color, default is 200.
    :return: Index of edge positions.
    """
    check_percentage = 0.03
    h, w, c = image.shape
    h_check = max(3, int(h * check_percentage)) # Check the number of consecutive light-colored rows
    w_check = max(3, int(w * check_percentage))
    # print("h_check, w_check")
    # print(h_check, w_check, h, w)
    mid_h, mid_w = h // 2, w // 2

    if direction == 'up':
        for i in range(mid_h - 1, -1, -1):
            if np.all(is_light_color(image[i, :, :], low_threshold)):
                # print('white line', i)
                is_edge = True
                if i == 0:
                    return 0
                for j in range(max(0, i - h_check), i):
                    if not np.all(is_light_color(image[j, :, :], low_threshold)):
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
    Remove the white borders of the image and fill the surrounding area with a specified size of white regions.
    :param image: Input image, in numpy array form, with shape (H, W, C).
    :param pad_size: The size of the surrounding padding, default is 5.
    :param low_threshold: The threshold for determining light colors, default is 200.
    :return: The processed image.
    """
    top = find_edge(image, 'up', low_threshold)
    bottom = find_edge(image, 'down', low_threshold)
    left = find_edge(image, 'left', low_threshold)
    right = find_edge(image, 'right', low_threshold)

    cropped_image = image[top:bottom, left:right]
    #
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

    # If it is (C, H, W), convert it to (H, W, C)
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    # Inverse Normalization: x = x * std + mean
    image = image * np.array(std) + np.array(mean)

    # Limit to [0, 1] to prevent error
    image = np.clip(image, 0, 1)
    return image


def convert_tensor_items_to_python_types(data):
    """
    Recursively traverse the elements in the dictionary or list, and convert the Tensors within them to Python data types.
    """
    if isinstance(data, dict):
        return {key: convert_tensor_items_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensor_items_to_python_types(element) for element in data]
    elif isinstance(data, torch.Tensor):
        # If Tensor is a scalar, use the item() method to convert it to a Python numeric type.
        if data.dim() == 0:
            return data.item()
        # Otherwise, convert the Tensor to a Python list
        return data.tolist()
    else:
        return data


def enhance_image_back(image_path, output_path):
    img = cv2.imread(image_path)

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using non-local means for denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Save the enhanced image
    cv2.imwrite(output_path, denoised)


def enhance_image(image_path, output_path):
    img = cv2.imread(image_path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    out_image = cv2.erode(gray_image, kernel, iterations=1)
    denoised = cv2.fastNlMeansDenoising(out_image, None, 10, 7, 21)
    # processed_image = np.where(denoised < 50, 0, denoised)

    cv2.imwrite(output_path, denoised)


def draw_molecule(transformed_image, atoms, bonds, save_path, input_size, json_path=None, smiles=None):
    img_vis = denormalize(transformed_image)
    # transformed_image = image
    # print(transformed_image.shape)  # (3, 384, 384)

    if transformed_image.shape[0] == 3 and len(transformed_image.shape) == 3:
        # Convert to (384, 384, 3)
        transformed_image = transformed_image.permute(1, 2, 0)

    img_width = input_size
    img_height = input_size

    # Convert the PIL image to an array format suitable for matplotlib.
    fig, ax = plt.subplots()
    ax.imshow(img_vis)
    ax.axis('off')  #

    # Set the font to facilitate the display of text.
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Adjust the font size according to the actual situation
    except IOError:
        font = ImageFont.load_default()

    # Defining the color of the key
    bond_colors = {'single': 'blue', 'double': 'green'}  # 

    # Draw atoms
    for atom in atoms:
        # print(atom['x']) # => Normalized coordinates
        x_pixel = atom['x'] * img_width  # Convert the normalized x-coordinate to pixel coordinates
        y_pixel = atom['y'] * img_height  # Convert the normalized y-coordinate to pixel coordinates
        ax.text(x_pixel, y_pixel, atom['atom_symbol'],
                color='red', fontsize=6, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="white", edgecolor="black", alpha=0.5))

    # Draw bonds
    for bond in bonds:
        start_atom = atoms[bond['endpoint_atoms'][0]]
        end_atom = atoms[bond['endpoint_atoms'][1]]
        ax.plot([start_atom['x'] * img_width, end_atom['x'] * img_width],
                [start_atom['y'] * img_height, end_atom['y'] * img_height],
                color=bond_colors.get(bond['bond_type'], 'red'), linewidth=2)  # The default red color is used for undefined key types.
        confidence_text_x = (start_atom['x'] + end_atom['x']) / 2 * img_width
        conficence_text_y = (start_atom['y'] + end_atom['y']) / 2 * img_height
        ax.text(confidence_text_x, conficence_text_y, '{:.3f}'.format(bond['confidence']), color='red', fontsize=6)

    # Save the image 
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()  # Close the image to free up memory

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
    Find the latest folder starting with 'molecules_detect_results_' in the given base directory based on timestamp.

    :param base_dir: The root directory path to search.
    :return: Full path of the latest folder found, None if no folder matches.
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
                    # Parse the timestamp string into a datetime object for comparison purposes.
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidate_folders.append((dt, item_path))
                except ValueError:
                    # The time format is invalid. Skipping.
                    continue

    if not candidate_folders:
        return None

    # Sort by timestamp and select the latest (maximum value)
    latest_folder = max(candidate_folders, key=lambda x: x[0])[1]
    return latest_folder


def main(opt):
    ckpt_path = opt.ckpt
    root_path = opt.dir2process
    all_pdfs_dir = os.listdir(root_path)
    skip_i = opt.skip_n
    print('Load the recognition model...')
    model = MolScribe(ckpt_path, device=torch.device('cpu'))
    model_states = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = get_args(model_states['args'])

    model_input_size = args.input_size
    # print( model_input_size) # => 384
    print('Model loading completed.')

    print(f"the total number of document folders{len(all_pdfs_dir)}")
    print(f"Startup file number：{skip_i}")
    ii = 0
    for pdf_dir_name in tqdm(all_pdfs_dir):
        if ii < skip_i:
            ii += 1
            continue

        obj_dir = os.path.join(root_path, pdf_dir_name)
        detect_outputs = find_latest_molecules_result_folder(obj_dir)
        if detect_outputs is None:
            print("No target detection result folder was found.")
            continue
        figure_dir = os.path.join(root_path, pdf_dir_name, detect_outputs, pdf_extracted_images_dir)
        if not os.path.isdir(figure_dir):
            print(f"No molecular folder was found:{figure_dir},skipping!")
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

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if with_prepocess:
                h_raw, w_raw = image.shape[:2]

                # Calculate the target size (as an integer)
                new_h = int(h_raw / ocr_expand_rate)
                new_w = int(w_raw / ocr_expand_rate)

                # Make sure not to go beyond the boundaries of the original image.
                new_h = min(new_h, h_raw)
                new_w = min(new_w, w_raw)

                # The starting coordinates for the calculation center clipping
                start_y = (h_raw - new_h) // 2
                start_x = (w_raw - new_w) // 2

                # clipping
                cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
                image = remove_white_borders(cropped, low_threshold=white_threshold)  # The white border remains at the same width as during training. pad = 5

            pred = model.predict_image(image, return_atoms_bonds=True, return_confidence=True)
            output = pred[0]
            images = pred[-1][0][0]
            # print(images.shape)

            smiles = output['smiles']
            all_mol_detected = smiles.split('.')
            longest_smiles = max(all_mol_detected, key=len)

            # print("All segments:", all_mol_detected)
            # print("The longest segment SMILES:", longest_smiles)
            # print("Length:", len(longest_smiles))

            save_image_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-raw-prediction.png")
            save_json_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-atoms-bonds.json")
            draw_molecule(images, output['atoms'], output['bonds'], save_image_path, model_input_size, json_path=save_json_path,
                          smiles=longest_smiles)

            if smiles:
                mol = Chem.MolFromSmiles(longest_smiles)
            else:
                pass

            if mol is None:
                # pass
                print("❌ RDKit parsing failed.")
                # Call the child process and correct the SMILES
                longest_smiles, im_path = run_mol_postprocess_and_get_smiles(save_json_path, output_log_dir)
                # Draw and save as PNG
                recognize_result_dict.update({im_name: {'smiles': longest_smiles,
                                                        'mol_fig_path': im_path}})

            else:

                # Optional:
                from rdkit.Chem import AllChem

                AllChem.Compute2DCoords(mol)

                img = Draw.MolToImage(mol, size=(400, 400), kekulize=True)
                output_img_path = os.path.join(output_log_dir, f"{base_name.split('.')[0]}-recognition.png")
                img.save(output_img_path)
                relative_image_path = os.path.relpath(output_img_path, output_dir)  # Save using relative path
                recognize_result_dict.update({im_name: {'smiles': longest_smiles,
                                                        'mol_fig_path': relative_image_path}})
                print(f"✅ Image {im_name} recognition succeed： {longest_smiles}")

        with open(result_json_path, 'w') as f:
            json.dump(recognize_result_dict, f, indent=4)
            f.close()


if __name__ == '__main__':
    run_opt = get_run_args()
    main(run_opt)

