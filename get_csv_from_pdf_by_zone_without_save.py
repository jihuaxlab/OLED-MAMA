import os
import fitz  # PyMuPDF
import camelot
import pandas as pd
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF 表格提取脚本，支持用户指定输入文件和输出目录")

    parser.add_argument(
        "-i", "--input",
        dest="input_pdf",
        required=True,
        help="输入 PDF 文件的路径"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output",
        required=True,
        help="输出目录路径（用于保存提取的表格和图像）"
    )

    parser.add_argument(
        "--csv_name",
        default=None,
        help="csv 名称，默认为pdf名称"
    )

    parser.add_argument(
        "-p", "--page",
        dest="page",
        required=False,
        type=int,
        default=None,
        help="抓取页码"
    )

    parser.add_argument(
        "--xn_start",
        type=float,
        default=None,
        help="抓取x起始位置")

    parser.add_argument(
        "--xn_end",
        type=float,
        default=None,
        help="抓取x结束位置")

    parser.add_argument(
        "--yn_start",
        type=float,
        default=None,
        help="抓取y起始位置")

    parser.add_argument(
        "--yn_end",
        type=float,
        default=None,
        help="抓取y结束位置")

    parser.add_argument('--upleft', action='store_true', help='xy start point')

    return parser.parse_args()


def normalize_to_actual(pdf_rect, xn_start, xn_end, yn_start, yn_end):
    rect = pdf_rect

    width = rect.width
    height = rect.height

    x_start = xn_start * width
    x_end = xn_end * width
    y_start = yn_start * height
    y_end = yn_end * height

    return int(x_start), int(x_end), int(y_start), int(y_end)


def extract_table_directly(pdf_path, page_num, bbox, output_csv):
    """
    直接在原始PDF的指定区域使用Camelot提取表格，避免fitz的字符问题
    """
    try:
        # 将bbox转换为Camelot需要的格式
        # Camelot使用字符串格式："x1,y1,x2,y2" (左下角到右上角)
        x1, y1, x2, y2 = bbox

        # 注意：Camelot的坐标系可能与fitz不同，可能需要调整
        area = f"{x1},{y2},{x2},{y1}"  # 调整y坐标


        # 直接使用stream方法（适用于无线表格）
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_num + 1),
            flavor='stream',
            table_areas=[area],
            strip_text='\n'
        )

        if len(tables) > 0:
            # 合并所有找到的表格
            dfs = [table.df for table in tables]
            if len(dfs) == 1:
                result_df = dfs[0]
            else:
                result_df = pd.concat(dfs, ignore_index=True)

            result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"✅ 表格已保存至：{output_csv}")
            return True
        else:
            print("❌ 未找到表格")
            return False

    except Exception as e:
        print(f"❌ 直接提取失败: {e}")
        return False


def ensure_within_bounds(page_rect, x_start, x_end, y_start, y_end):
    """
    Ensure that the coordinates are within the bounds of the page.
    """
    x_start = max(0, min(page_rect.x1, x_start))
    x_end = max(0, min(page_rect.x1, x_end))
    y_start = max(0, min(page_rect.y1, y_start))
    y_end = max(0, min(page_rect.y1, y_end))

    return x_start, x_end, y_start, y_end
