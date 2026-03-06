import os
from dashscope import MultiModalConversation
import dashscope
import json
import sys
import re

KEYS_ = 'your KEYS_' # JHL
# KEYS_ =  "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# If using the Singapore region, please remove the comment on the next line (select based on the region associated with your API Key)
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

def extract_page_number(filename):
  match = re.search(r'page_(\d+)', filename)
  return int(match.group(1)) if match else None


def group_consecutive_pages(page_numbers, n_max=1):
    """Group the sorted page number list into consecutive intervals"""
    if not page_numbers:
        return []
    
    page_numbers = sorted(set(page_numbers))
    groups = []
    start = end = page_numbers[0]
    count = 1  # The current group page numbers.

    for num in page_numbers[1:]:
        # Check if it is continuous and the limit.
        if num == end + 1 and count < n_max:
            end = num
            count += 1
        else:
            groups.append((start, end))
            start = end = num
            count = 1

    groups.append((start, end))
    return groups

table_imgs_dir = sys.argv[1]
print("Table image file folder path:", table_imgs_dir)

if not os.path.isdir(table_imgs_dir):
    print('ERROR: The folder for the table image does not exist.')
    sys.exit(1)

png_files = []
for fname in os.listdir(table_imgs_dir):
    if fname.lower().endswith('.png'):
        page_num = extract_page_number(fname)
        if page_num is not None:
            png_files.append((page_num, fname))

if not png_files:
    print("未找到任何符合命名规范的 PNG 文件")
    sys.exit(0)

# Sort by page number
png_files.sort(key=lambda x: x[0])

# Extract all single-page numbers and group them into consecutive intervals.
all_pages = [p for p, _ in png_files]
page_groups = group_consecutive_pages(all_pages)

# print(page_groups)

# mapping from page numbers to the list of files
page_to_files = {}
for page_num, fname in png_files:
    page_to_files.setdefault(page_num, []).append(fname)


# Construction prompt: Clearly specify that the response should be in the specified JSON format.
# The English version in the Supporting Information of the paper can be used.
prompt = """
请仔细分析并提取图中的表格数据，并按照以下JSON格式输出：

{
  "All_condition": {
    "a": "例如 measured in MeCN at 298K.",
    "b": "例如 measurements in air.",
    ...
  },
  "All_Mol": {
    "分子名称代号1": {
      "λem [nm]": [{"value": 数值, "condition": "条件键"}, ...],
      "Φ PLQY [%]": [{"value": 数值, "condition": "条件键"}, ...],
      "t(PF ns / DF μs)": [{"value": [PF值, DF值], "condition": "条件键"}, ...],
      "CIE": [{"value": [数值, 数值], "condition": "条件键"}, ...], 
      "k_r (× 10⁷ s⁻¹)": [{"value": 数值, "condition": "条件键"}, ...],
      "... 其他属性(D_EST [eV], E_S [eV], E_T [eV]等表格中所有属性)都是用测试结果的list形式列出... "
      ...
    },
    "分子名称代号2": {...},
    ...
  }
}

注意：
1. 条件（如 a, b, c...）需从表格标题或注释中提取，并统一编号。
2. 重点关注Φ PLQY [%], EQE [%], λem [nm], λabs [nm], D_EST [eV](也就是DeltaE_ST), E_S [eV], E_T [eV] 和 速率k等数据，数据的单位需要放置在键的名称的括号中，有时候表格没注明Φ PLQY单位时，如果所有的PLQY数值都小于1则说明是没有添加百分数单位，因此在提取数据时需要统一乘以100。尽可能记录表格中出现的所有属性名称和数值，没有数据的属性键则需要省略。
3. 所有数值保留原始精度，不要四舍五入。
4. 如果无法识别某部分，或者表格数据有空缺的地方，用下面方式进行填充，不要重复和编造（重要）。
   - 单值字段（如 Φ PLQY）→ `"value": "-"`
   - 双值字段（如 t(PF/DF) 或 CIE）→ `"value": ["-", "-"]`
   - 条件缺失 → `"condition": "-"`
5. Emitter, EML, Host 的详情需要计入condition中，例如condition： "a, mCBP:5% emitter 名称"
`"λabs(nm)`": [
{
`"value`": [285, 316, 466],
`"condition`": `"a, Cyclohexane`"
}
6. 最终输出必须是合法JSON，不要包含任何额外文字或解释。
7. 表格中可能有些地方数据空缺，必须保证分子名称与数据行严格对齐，不可错行，不要编造（重要）。
8. 如果图像没有提取到分子名称和属性数据则输出空的json {"All_condition": {}, "All_Mol": {}}。
9. 如果图像中包含EQE，CIE等说明是器件数据的情况，需要在分子名称后添加 device 进行说明，例如 "分子名称代号 device"。
10. 如果condition的格式为用\\或者/隔开的多个条件（例如 sol/film或者 THF/doped film等），并和value形成一一对应的关系，尽可能将其进行拆分记录，PLQY [%]": [{"value": 91,"condition": "sol"},{"value": 99,"condition": "film"}]
"""

# Process each consecutive group of page numbers
for start_page, end_page in page_groups:
    # 
    image_content = []
    involved_pages = range(start_page, end_page + 1)
    has_files = False
    for p in involved_pages:
        if p in page_to_files:
            for fname in page_to_files[p]:
                fpath = os.path.join(table_imgs_dir, fname)
                if os.path.isfile(fpath):
                    image_path = f"file://{os.path.abspath(fpath)}"
                    image_content.append({'image': image_path})
                    has_files = True
    if not has_files:
        continue

    # Construct a complete prompt
    image_content.append({'text': prompt})
    full_prompt = image_content

    messages = [
        {
            'role': 'user',
            'content': full_prompt,
        }
    ]
    
    # print(messages)
    # raise 

    # Using the LLM model
    try:
        response = MultiModalConversation.call(
            # api_key=os.getenv('DASHSCOPE_API_KEY'),  # Make sure that the environment variables have been set.
            api_key=KEYS_,  # Make sure that the environment variables have been set.
            model='qwen3-vl-plus',  # u can use the model that you prefer:
            # model='qwen3-vl-plus-2025-12-19',
            messages=messages,
            extra_body={"vl_high_resolution_images": True,
                        'enable_thinking': True,
                        "thinking_budget": 81920
                        },
            temperature=0.01  # Reduce randomness and enhance the stability of structured output
        )

        output_text = response.output.choices[0].message.content[0]["text"]

        # Try to parse as JSON (the model may return in a format with ```json...``` )
        if output_text.strip().startswith("```json"):
            json_str = output_text.strip()[7:-3]  # Remove ```json and ```
        else:
            json_str = output_text.strip()

        # print(json_str)

        result_json = json.loads(json_str)
        print('JSON loading successful')
        # print(json.dumps(result_json, indent=2, ensure_ascii=False))
        # Save the results
        if start_page == end_page:
            output_name = f"tongyi_response_page{start_page}.json"
        else:
            output_name = f"tongyi_response_page{start_page}-{end_page}.json"

        output_path = os.path.join(table_imgs_dir, output_name)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            print(f"✅ Saved at: {output_name}")
        except Exception as e:
            print(f"❌ Save failed: {e}")

    except Exception as e:
        print(f"API failed（Page {start_page}-{end_page}）: {e}")
        continue
    # raise
