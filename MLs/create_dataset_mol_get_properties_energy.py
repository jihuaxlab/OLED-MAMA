# 用于指定的属性构建数据集
# 细节：按顺序遍历所有paper文件夹中的json：基于json中记录的数据按照（paper，分子名称）寻找MOLID
# 用于提取能量相关属性
import os
import json
import pandas as pd
import re
from collections import defaultdict
import csv
import statistics

# -----------------------------
# 配置路径
# -----------------------------
# 合集
MAPPING_CSV = r'./data\smiles_to_id_mapping_12.csv'
PAPER_MAPPING_CSV = r'/data\output_mapping_12.csv'
PAPER_ROOT_DIR = r'your pdf output dir' # 
OUTPUT_DIR = r'./output'

OUTPUT_FILE_NAME = 'mol_energy_data_with_condition.csv'
# OUTPUT_FILE_NAME = 'mol_energy_data_simulation.csv'

# -----------------------------
# 属性定义（支持 | 分隔的别名）
# -----------------------------
# obj_properties_raw = [
#     'E_ST|D_ST|EST', 'λem', 'λ_EL|λEL', 'PLQY|', 'S₁|S1', 'T₁|T1', 'k_rISC|krISC|k_RISC|k_risc',
#     'EQE|External Quantum Efficiency', 'CIE', 'FWHM', 'PEmax|lm W⁻¹|lm W-1|Power efficiency', 'CEmax|cd A¹|cd A-1|Current efficiency'
# ]

obj_properties_raw = [
    'E_ST|D_ST|EST|E_S-T|E_S1-T1', 'S₁|S1|Es/|ES/|E_S |E_S0-S1|Singlet energy', 'T₁|T1|Et/|ET/|E_T |ET |E_T0-T1|Triplet energy',
]

exclude_symbol = ['→', 'S₀', 'S0', 'S₂', 'S2', 'T₀', 'TO', 'T₂', 'T2', 'S₁-T₁', 'S1-T1', 'T₁-S₁', 'T1-S1', 'S1-T3', 'S1-T2']

# first_selection = ['甲苯', 'toluene', 'methylbenzene', 'solution', 'dilute']  # 优先选择甲苯溶液中的数据
first_selection = ['neat film', 'toluene', 'PMMA', 'mCP', 'mCBP', 'DPEPO', '2-MeTHF', 'THF', 'PPT', 'PPF',
                   'zeonex', 'CBP', 'DBFPO', 'mDCBP', 'BCPO', 'mCPCN', 'CH2Cl2', 'CHCl3', 'MeCN', 'PLA',
                   '2,6-DCzPPy', 'TCTA', 'PO-T2T', 'B3PYMPM', 'DOBNA-OAr', 'Bepp2', 'ethanol', 'cyclohexane',
                   'hexane', 'SiTrzCz2',
                   'pyd2', 'CzSi', 'PhCzBCz', 'PS', 'TSPO1', 'TPBi']  # 可优先选择甲苯溶液中的数据
# first_selection = ['simulation']  # DFT

condtion = {'PMMA' :['PMMA', 'polymethyl methacrylate'],
            'mCP': ['mCP', '1,3-Bis(N-carbazolyl)benzene', '1,3-di-9-carbazolylbenzene'],
            'mCBP': ['mCBP'],
            'DPEPO': ['DPEPO', 'Bis[2-(diphenylphosphino)phenyl]ether oxide'],
            'PTT': ['Polythienothiophene', 'PTT'],
            'PPF': ['PPF'],
            'PPT': ['PPT', '2,8-bis-(diphenylphosphoryl)dibenzo[b,d]thiophene'],
            'zeonex': ['zeonex'],
            'CBP': ['CBP'],
            'DBFPO': ['DBFPO'],
            '2-MeTHF': ['2-MeTHF', '2-methyltetrahydrofuran', 'Me-THF'],
            'THF': ['THF', 'tetrahydrofuran'],
            '2-Methylfuran': ['2-Methylfuran'],
            'mDCBP': ['mDCBP'],
            'BCPO': ['BCPO'],
            'mCPCN': ['mCPCN'], #  (3,5-di(9H-carbazol-9-yl)phenyl)(pyridin-4yl)
            'CH2Cl2': ['CH₂Cl₂', 'CH2Cl2', 'dichloromethane', 'DCM'],
            'CHCl3': ['CHCl₃', 'CHCl3', 'Chloroform'],
            'MeCN': ['MeCN', 'Acetonitrile'],
            'toluene': ['甲苯', 'toluene', 'methylbenzene', 'Tol.'],
            'PLA': ['PLA', 'poly(lactic acid)'],
            '2,6-DCzPPy': ['2,6-DCzPPy'],
            'TCTA': ['TCTA'],
            'SiTrzCz2': ['SiTrzCz2'],
            'cyclohexane': ['cyclohexane'],
            'hexane': ['hexane'],
            'ethanol': ['ethanol', 'EtOH'],
            'Bepp2': ['Bepp2', 'Bepp₂'],
            'DOBNA-OAr': ['dobna-oar'],
            'B3PYMPM': ['B3PYMPM'],
            'PO-T2T': ['PO-T2T'],
            'o-dcb': ['o-dcb', '1,2-dichlorobenzene'],
            'pyd2': ['pyd2'],
            'CzSi': ['CzSi'],
            'PhCzBCz': ['PhCzBCz'],
            'PS': ['PS', 'polystyrene'],
            'TSPO1': ['TSPO1'],
            'TPBi': ['TPBi'],
            # 'liquid nitrogen': ['liquid nitrogen'],

            'neat film': ['neat film', 'neat', 'non-doped film', 'solid-state film', 'solid state', 'liquid nitrogen'], # the film state 得参考
            'simulation': ['calculated', 'Cal.', 'Gauss', 'simulation', 'simulated', 'DFT', 'B3LYP', 'PBE0', 'Theoretical']}


# 手动构建：标准名称 -> SMILES
name_to_smiles = {
    'PMMA': 'CCC(C)(C)C(=O)OC',
    'mCP': 'c1cc(-n2c3ccccc3c3ccccc32)cc(-n2c3ccccc3c3ccccc32)c1',
    'mCBP': 'c1cc(-c2cccc(-n3c4ccccc4c4ccccc43)c2)cc(-n2c3ccccc3c3ccccc32)c1',
    'DPEPO': 'O=P(c1ccccc1)(c1ccccc1)c1ccccc1Oc1ccccc1P(=O)(c1ccccc1)c1ccccc1',
    'PPF': 'O=P(c1ccccc1)(c1ccccc1)c1ccc2oc3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3c2c1',
    'PPT': 'O=P(c1ccccc1)(c1ccccc1)c1ccc2sc3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3c2c1',
    'zeonex': 'CCC1CC(CC)C2C3CCC(C3)C12',
    'CBP': 'c1ccc2c(c1)c1ccccc1n2-c1ccc(-c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1',
    'DBFPO': 'O=P(c1ccccc1)(c1ccccc1)c1ccc2oc3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3c2c1',
    'THF': 'C1CCOC1',
    '2-MeTHF': 'CC1CCCO1',
    '2-Methylfuran': 'CC1=CC=CO1',
    'mDCBP': 'O=C(C1=CC=NC=C1)C2=CC(N3C4=C(C=CC=C4)C5=C3C=CC=C5)=CC(N6C7=C(C=CC=C7)C8=C6C=CC=C8)=C2',
    'BCPO': 'c1ccc(P(c2ccc(-n3c4ccccc4c4ccccc43)cc2)c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1',
    'mCPCN': 'N#Cc1ccc2c(c1)c1ccccc1n2-c1cccc(-n2c3ccccc3c3ccccc32)c1',
    'CH2Cl2': 'ClCCl',
    'CHCl3': 'ClC(Cl)Cl',
    'MeCN': 'CC#N',
    'toluene': 'CC1=CC=CC=C1',
    'PLA': 'CC(C(O)=O)O',
    '2,6-DCzPPy': 'c1cc(-c2cccc(-c3cccc(-n4c5ccccc5c5ccccc54)c3)n2)cc(-n2c3ccccc3c3ccccc32)c1',
    'TCTA': 'c1ccc2c(c1)c1ccccc1n2-c1ccc(N(c2ccc(-n3c4ccccc4c4ccccc43)cc2)c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1',
    'SiTrzCz2': 'c1ccc([Si](c2ccccc2)(c2ccccc2)c2cccc(-c3nc(-n4c5ccccc5c5ccccc54)nc(-n4c5ccccc5c5ccccc54)n3)c2)cc1',
    'hexane': 'CCCCCC',
    'ethanol': 'CCO',
    'Bepp2': 'c1ccc2c(c1)O[B-]1(Oc3ccccc3-c3cccc[n+]31)[n+]1ccccc1-2',
    'DOBNA-OAr': 'Cc1ccccc1-c1cccc(Oc2cc3c4c(c2)Oc2cc(-c5ccccc5C)ccc2B4c2ccc(-c4ccccc4C)cc2O3)c1',
    'B3PYMPM': 'CC1=NC(C2=CC(C3=CN=CC=C3)=CC(C4=CC=CN=C4)=C2)=CC(C5=CC(C6=CC=CN=C6)=CC(C7=CC=CN=C7)=C5)=N1',
    'PO-T2T': 'O=P(c1ccccc1)(c1ccccc1)c1cccc(-c2nc(-c3cccc(P(=O)(c4ccccc4)c4ccccc4)c3)nc(-c3cccc(P(=O)(c4ccccc4)c4ccccc4)c3)n2)c1',
    'o-dcb': 'ClC1=CC=CC=C1Cl',
    'pyd2': 'c1cc(-n2c3ccccc3c3ccccc32)nc(-n2c3ccccc3c3ccccc32)c1',
    'CzSi': 'CC(C)(C)c1ccc(-n2c3ccc([Si](c4ccccc4)(c4ccccc4)c4ccccc4)cc3c3cc([Si](c4ccccc4)(c4ccccc4)c4ccccc4)ccc32)cc1',
    'PhCzBCz': 'c1ccc(-n2c3ccccc3c3cc(-c4ccccc4-n4c5ccccc5c5cc(-n6c7ccccc7c7ccccc76)ccc54)ccc32)cc1',
    'PS': 'CC(CC(C1=CC=CC=C1)CC)C2=CC=CC=C2',
    'TSPO1': 'O=P(c1ccccc1)(c1ccccc1)c1ccc([Si](c2ccccc2)(c2ccccc2)c2ccccc2)cc1',
    'TPBi': 'c1ccc(-n2c(-c3cc(-c4nc5ccccc5n4-c4ccccc4)cc(-c4nc5ccccc5n4-c4ccccc4)c3)nc3ccccc32)cc1',
}

host_smiles = ['O=P(c1ccccc1)(c1ccccc1)c1cccc(-c2nc(-c3cccc(P(=O)(c4ccccc4)c4ccccc4)c3)nc(-c3cccc(P(=O)(c4ccccc4)c4ccccc4)c3)n2)c1', # PO-T2T
               'c1ccc2c(c1)c1ccccc1n2-c1ccc(-c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1', # CBP
               'c1cc(-c2cccc(-n3c4ccccc4c4ccccc43)c2)cc(-n2c3ccccc3c3ccccc32)c1', # mCBP
               'O=P(c1ccccc1)(c1ccccc1)c1ccccc1Oc1ccccc1P(=O)(c1ccccc1)c1ccccc1', # DPEPO
               'c1cc(-n2c3ccccc3c3ccccc32)cc(-n2c3ccccc3c3ccccc32)c1', # mCP
               'O=P(c1ccccc1)(c1ccccc1)c1ccc2oc3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3c2c1', # PPF
               'c1ccc2c(c1)c1ccccc1n2-c1ccc(N(c2ccc(-n3c4ccccc4c4ccccc43)cc2)c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1', # 'TCTA'
               'ClCCl', # CCL2 二氯甲烷 DCM
               'ClC(Cl)Cl', # CCL3
               'ClC(Cl)(Cl)Cl', # CCL4 CTC
               'CCC(C)(C)C(=O)OC', # PMMA
               'O=P(c1ccccc1)(c1ccccc1)c1ccc2sc3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3c2c1',  # PPT
               'N#Cc1cc(-c2cccc(-n3c4ccccc4c4ccccc43)c2)cc(-n2c3ccccc3c3ccccc32)c1', # mCBPCN
               'CC(C)(C)c1ccc(-n2c3ccc([Si](c4ccccc4)(c4ccccc4)c4ccccc4)cc3c3cc([Si](c4ccccc4)(c4ccccc4)c4ccccc4)ccc32)cc1', # CzSi
               'Cc1ccccc1', # Toluene
               'CCC1CC(CC)C2C3CCC(C3)C12', # ZEONEX
               'N#Cc1ccc2c(c1)c1ccccc1n2-c1cccc(-n2c3ccccc3c3ccccc32)c1', # mCPCN
               'C1CCOC1', # THF
               'C1CCOCC1', # THP
               'CC(C)(C)c1ccc(Oc2ccc(C(C)(C)C)cc2P(=O)(c2ccccc2)c2ccccc2)cc1', # POBBPE
               'C1CCCCC1', # Cyclohexane
               'c1ccc(-n2c(-c3cc(-c4nc5ccccc5n4-c4ccccc4)cc(-c4nc5ccccc5n4-c4ccccc4)c3)nc3ccccc32)cc1',  # TPBi
               'O=P(c1ccccc1)(c1ccccc1)c1ccc2c(c1)c1ccccc1n2-c1cccc(-n2c3ccccc3c3ccccc32)c1',  # mCPPO
               'CC(C)(C)c1ccc(-n2c3ccc([Si](c4ccccc4)(c4ccccc4)c4ccccc4)cc3c3cc([Si](c4ccccc4)(c4ccccc4)c4ccccc4)ccc32)cc1', # CzSi
               'c1cc(-c2cccc(-c3cccc(-n4c5ccccc5c5ccccc54)c3)n2)cc(-n2c3ccccc3c3ccccc32)c1', # 2,6-DCzppy
               'CCC(C)n1c2ccccc2c2ccccc21', # PVK
               'Cc1cc(-n2c3ccccc3c3ccccc32)cc(C)c1C1c2ccccc2C(c2c(C)cc(-n3c4ccccc4c4ccccc43)cc2C)c2ccccc21', # CzDBA
               'Cc1cccc(N(c2ccccc2)c2ccc(N(c3ccc(N(c4ccccc4)c4cccc(C)c4)cc3)c3ccc(N(c4ccccc4)c4cccc(C)c4)cc3)cc2)c1', # m-MTDATA
               'CC(C)(C)c1ccc(-c2nnc(-c3cccc(-c4nnc(-c5ccc(C(C)(C)C)cc5)o4)c3)o2)cc1', # OXD-7
               'CC(C)(C)c1ccc2c(c1)c1cc(C(C)(C)C)ccc1n2-c1cc(-n2c3ccc(C(C)(C)C)cc3c3cc(C(C)(C)C)ccc32)c(-n2c3ccccc3c3ccccc32)c(C#N)c1-n1c2ccccc2c2ccccc21', # 2tCz2CzBn
               'CN1c2ccc3cc2C(C)(C)c2cc(ccc21)Cc1ccc2c(c1)C(C)(C)C1CC(CCC1N2C)Cc1ccc2c(c1)C(C)(C)c1cc(ccc1N2C)C3', # C[3]A
               'O=P(c1ccccc1)(c1ccccc1)c1ccc2oc3ccc(P(=O)(c4ccccc4)c4ccccc4)cc3c2c1', # DBFPO
               'c1ccc2c(c1)O[B-]1(Oc3ccccc3-c3cccc[n+]31)[n+]1ccccc1-2', # BEPP2
               'Cc1nc(-c2cc(-c3ccncc3)cc(-c3ccncc3)c2)cc(-c2cc(-c3ccncc3)cc(-c3ccncc3)c2)n1', # B4PyMPM
               'c1cc(-n2c3ccccc3c3ccccc32)nc(-n2c3ccccc3c3ccccc32)c1', # PYD2
               'c1ccc(P(c2ccc(-n3c4ccccc4c4ccccc43)cc2)c2ccc(-n3c4ccccc4c4ccccc43)cc2)cc1', # BCPO
               'c1cc(-c2cccc(-c3cccc(-n4c5ccccc5c5ccccc54)c3)n2)cc(-n2c3ccccc3c3ccccc32)c1', # 26DCzPPy
               'CC1(C)c2ccccc2-c2cc3c4ccccc4n(-c4cccc(-c5nc(-c6ccccc6)nc(-c6ccccc6)n5)c4)c3cc21', # DMIC-TRZ
               'c1ccc(-n2c3ccccc3c3cc(-c4ccccc4-n4c5ccccc5c5cc(-n6c7ccccc7c7ccccc76)ccc54)ccc32)cc1', # PhCzBCz
               'c1ccc([Si](c2ccccc2)(c2ccccc2)c2cccc(-c3nc(-n4c5ccccc5c5ccccc54)nc(-n4c5ccccc5c5ccccc54)n3)c2)cc1', # SiTrzCz2
               'c1ccc([Si](c2ccccc2)(c2ccccc2)c2cccc(-n3c4ccccc4c4cc(-n5c6ccccc6c6ccccc65)ccc43)c2)cc1', # SiCzCz
               'O=S1(=O)c2ccccc2C(c2ccc3c(c2)c2ccccc2n3-c2ccccc2)(c2ccc3c(c2)c2ccccc2n3-c2ccccc2)c2ccccc21', # BPhCz-ThX
               'O=P(c1ccccc1)(c1ccccc1)c1ccc([Si](c2ccccc2)(c2ccccc2)c2ccccc2)cc1', # TSPO1
               ]

# 解析为：prop_name -> [keyword1, keyword2, ...]
property_keywords = {}
for item in obj_properties_raw:
    parts = [p.strip() for p in item.split('|') if p.strip()]
    if not parts:
        continue
    main_name = parts[0]  # 用第一个作为代表名
    property_keywords[main_name] = parts

print("属性关键词映射:")
for k, v in property_keywords.items():
    print(f"  {k}: {v}")

# -----------------------------
# 步骤1: 读取论文-分子映射表，构建 (paper_folder, mol_name) -> set(molecule_ids)
# -----------------------------
paper_df = pd.read_csv(PAPER_MAPPING_CSV, encoding='gbk')
name_to_ids = {}  # key: (paper_folder, mol_name), value: set of Molecule_ID

for _, row in paper_df.iterrows():
    paper_folder = str(row['PAPER']).strip()
    mol_name = str(row['Name']).strip()
    ids_str = str(row['Molecule_IDs']).strip()

    if ids_str == '' or ids_str.lower() == 'nan':
        continue

    id_list = [x.strip() for x in ids_str.split('\\') if x.strip()]
    if id_list:
        name_to_ids[(paper_folder, mol_name)] = set(id_list)

print(f"\n共加载 {len(name_to_ids)} 个 (论文, 分子代号) 映射项")

id_smiles_df = pd.read_csv(MAPPING_CSV, encoding='gbk')
ids_to_smiles = {}

for _, row in id_smiles_df.iterrows():
    mol_id = str(row['Molecule_ID']).strip()
    smiles_str = str(row['SMILES']).strip()

    if mol_id == '' or smiles_str.lower() == 'nan':
        continue

    if mol_id:
        ids_to_smiles[mol_id] = smiles_str

print(f"\n共加载 {len(ids_to_smiles)} 个 (分子代号-smiles) 映射项")
id_to_smiles_names = list(ids_to_smiles.keys())

# -----------------------------
# 步骤2: 初始化统计字典
# -----------------------------
property_mol_sets = defaultdict(set)  # prop_name -> set(Molecule_ID)

# -----------------------------
# 步骤3: 遍历每篇论文，只读取其最新 molecules_detect_resultsXXXX/table-images/ 下的 JSON
# -----------------------------
print("\n开始解析各论文最新结果中的 JSON...")

# 获取所有唯一的 paper_folder 名称
unique_papers = set(paper_df['PAPER'].dropna().astype(str).str.strip())
# unique_papers = paper_df['PAPER'].dropna().astype(str).str.strip()
unique_papers = list(unique_papers)
unique_papers.sort()
processed_count = 0

skip_i = 0

from collections import defaultdict
final_data = defaultdict(lambda: defaultdict(list))  # mol_id -> {prop: [values]}

for paper_name in unique_papers:
    if processed_count < skip_i:
        processed_count += 1
        continue
    if paper_name.lower() in ('', 'nan'):
        continue

    paper_full_path = os.path.join(PAPER_ROOT_DIR, paper_name)
    if not os.path.isdir(paper_full_path):
        # 尝试匹配部分名称？或跳过
        print(f"警告: 论文文件夹不存在: {paper_full_path}")
        continue

    # 进入 table-images 子目录
    table_images_dir = os.path.join(paper_full_path, "table-images")
    if not os.path.isdir(table_images_dir):
        print(f"跳过: {paper_name} 的 table-images 目录不存在")
        continue

    # 查找该目录下所有 .json 文件
    json_files = [f for f in os.listdir(table_images_dir) if f.endswith('.json')]
    if not json_files:
        print(f"跳过: {paper_name} 的 table-images 中无 JSON 文件")
        continue

    # 解析每个 JSON
    for json_file in json_files:
        json_path = os.path.join(table_images_dir, json_file)
        print(f"加载 JSON: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"跳过损坏 JSON: {json_path} | 错误: {e}")
            continue

        all_mol = data.get("All_Mol", {})
        all_condition = data.get("All_condition", {})
        # all_device = data.get("All_Device", {}) # 后续可能会有
        if not isinstance(all_mol, dict):
            continue

        for mol_name, props in all_mol.items():
            if not isinstance(props, dict):
                continue

            # 构造 key = (paper_name, mol_name)
            key = (paper_name, str(mol_name).strip())
            if key not in name_to_ids:
                # 可选：打印未匹配项用于调试
                print(f"警告: 未找到映射 ({paper_name}, {mol_name})")
                # raise
                continue
            else:
                pass
                # print(f"到映射: ({paper_name}, {mol_name})")

            molecule_ids = name_to_ids[key]

            # 过滤出有 SMILES 的有效 ID
            valid_mol_entries = []
            for mid in molecule_ids:
                if mid in id_to_smiles_names:
                    can_smi = ids_to_smiles[mid]
                    if can_smi:
                        valid_mol_entries.append((mid, can_smi))

            if not valid_mol_entries:
                continue  # 无有效 SMILES，跳过

            # 分组：是否在 host_smiles 中
            in_host = []
            not_in_host = []
            for mid, smi in valid_mol_entries:
                if smi in host_smiles:
                    in_host.append((mid, smi))
                else:
                    not_in_host.append((mid, smi))

            # 选择策略
            if len(in_host) == len(valid_mol_entries) and in_host:
                # 所有都在 host 中 → 选 SMILES 最长的
                selected_mid, _ = max(in_host, key=lambda x: len(x[1]))
            elif not_in_host:
                # 有非 host → 从非 host 中选最长的
                selected_mid, _ = max(not_in_host, key=lambda x: len(x[1]))
            else:
                # # 没有分子理论上不会发生
                print("⚠ 没有分子")
                continue


            # 检查属性匹配
            for json_key, value in props.items():
                # Step 1: 找出这个 json_key 匹配了哪些目标属性（按顺序！）
                matched_props_with_pos = []
                for prop_name, keywords in property_keywords.items():
                    min_pos = float('inf')
                    found = False
                    for kw in keywords:
                        if not kw:
                            continue
                        pos = json_key.find(kw)
                        if pos != -1:
                            found = True
                            if pos < min_pos:
                                min_pos = pos
                    if found:
                        matched_props_with_pos.append((min_pos, prop_name))

                # 按出现位置排序
                matched_props_with_pos.sort(key=lambda x: x[0])
                matched_props = [prop for _, prop in matched_props_with_pos]
                if len(matched_props) == 0:
                    continue

                # === 新增：从 json_key 中提取单位 ===
                json_key_lower = json_key.lower()
                if 'mev' in json_key_lower or '[mev]' in json_key_lower or ' mev' in json_key_lower or r'/mev' in json_key_lower:
                    unit = 'meV'
                elif '(ev)' in json_key_lower or '[ev]' in json_key_lower or ' ev' in json_key_lower or r'/ev' in json_key_lower:
                    unit = 'eV'
                else:
                    unit = None

                if unit is None:
                    # 和能量单位不匹配，跳过
                    continue

                need_pass = False
                for ele in exclude_symbol:
                    if ele in json_key:
                        need_pass = True
                        break
                if need_pass:
                    continue

                # raw_unit = unit
                # 可扩展：处理 nm, %, cd/A² 等（当前仅关注能量单位）
                print(f'数据：{paper_name} - {mol_name} --> {json_key} : {value}')

                # 尝试提取数值（支持列表或单个值）
                values_to_process = value if isinstance(value, list) else [value]
                value_ = None
                has_priority = False
                data_dict = {}
                ok_ = False
                for item in values_to_process:
                    if isinstance(item, dict):
                        v_str = item.get('value', '')
                        if v_str and value_ is None:
                            value_ = v_str
                        cond_str = str(item.get('condition', '')).strip()
                        if '/' in cond_str or '\\' in cond_str:
                            print("多条件: ", cond_str)

                        # Step 1: 检查是否满足 first_selection（高优先级）
                        cond_lower = cond_str.lower()
                        print("记录条件：", cond_lower)
                        has_priority = False
                        test_condition = None
                        for selection_key in first_selection:
                            has_priority = any(sel.lower() in cond_lower for sel in condtion[selection_key])
                            if has_priority:
                                print(f"匹配测试条件{selection_key}")
                                test_condition = selection_key
                                break

                        if not has_priority:
                            # Step 2: 没有优先条件 → 检查子条件是否在 All_condition 中有效
                            if cond_str:
                                # 拆分条件，支持 "g, h"、"g; h"、"g h" 等
                                sub_conds = re.split(r'[,;，；\s]+', cond_str)
                                sub_conds = [c.strip() for c in sub_conds if c.strip()]

                                # 检查是否有至少一个子条件在 all_condition 中存在且非空
                                for sc in sub_conds:
                                    if sc in all_condition:
                                        # 可选：进一步检查 all_condition[sc] 是否有效（非空、非 None）
                                        cond_info = all_condition[sc]
                                        print("检查子条件： ", cond_info)
                                        if cond_info not in (None, '', {}, [], 'N/A'):
                                            cond_lower = cond_info.lower()
                                            for selection_key in first_selection:
                                                has_priority = any(
                                                    sel.lower() in cond_lower for sel in condtion[selection_key])
                                                if has_priority:
                                                    print(f"匹配测试条件{selection_key}")
                                                    test_condition = selection_key
                                                    break
                                            if has_priority:
                                                break
                        if has_priority:
                            value_ = v_str
                            if value_ is None:
                                continue
                            # 转为字符串处理
                            if isinstance(value_, str):
                                value_ = str(value_).strip()
                                # 情况 A: "425/—" → 拆分为 ["425", "—"]
                                if '/' in value_ and not value_.startswith('http'):  # 避免误拆 URL
                                    parts = [p.strip() for p in value_.split('/')]
                                else:
                                    parts = [value_]
                            # 情况 B: [2.91, 2.82] → 已是列表
                            elif isinstance(value_, list):
                                parts = [str(x).strip() for x in value_]
                            else:
                                # 单个值
                                parts = [value_]

                            # 尝试转 float，失败则跳过
                            nums = []
                            for p in parts:
                                if p in ('—', '-', 'N/A', '', 'None'):
                                    nums.append(None)
                                else:
                                    try:
                                        nums.append(float(p))
                                    except ValueError:
                                        nums.append(None)

                            # 如果没有有效数值，跳过
                            if not any(x is not None for x in nums):
                                print('无有效数值', nums)
                                print()
                                continue

                            # Step 4: 将 flat_values 按顺序分配给 matched_props
                            # 例如: matched_props = ['S1', 'T1'], flat_values = [2.91, 2.82]
                            for i, prop_name in enumerate(matched_props):
                                if i >= len(nums):
                                    break
                                val = nums[i]
                                if val is None:
                                    print()
                                    continue

                                # === 单位转换逻辑 ===
                                if unit == 'eV':
                                    val = val * 1000.0  # 转为 meV
                                    # print(f"  [Unit] {json_key} -> {num} meV")
                                elif unit == 'meV':
                                    pass  # 保持不变
                                else:
                                    # 无单位或未知单位：可选择跳过、警告、或默认视为 meV
                                    # 这里我们保守处理：**仅当属性是能量类时才要求单位**
                                    # 对于 E_ST/S1/T1，建议强制要求单位，否则跳过
                                    print(paper_name, ' : ', json_file)
                                    print(f"⚠️ 能量属性 '{prop_name}' 在 key='{json_key}' 中未识别单位，跳过值: {val}")
                                    print()
                                    continue
                                    # 非能量属性（如未来扩展 λem）可保留原始值
                                print(f"🆗️ 能量属性 '{prop_name}' 在 key='{json_key}' 中, 值: {val}")
                                data_dict = {'val': val, 'condition': test_condition}
                                print(f"提取属性 '{prop_name}", data_dict)
                                final_data[selected_mid][prop_name].append(data_dict)
                                if not ok_:
                                    ok_ = True
                        else:
                            print('当前数据： ', v_str, '是否满足首选测试条件：', has_priority)
                    else:
                        # list 或数值
                        # if item and value_ is None:
                        #     value_ = item
                        print("数据数据无效")

                print('当前分子： ', mol_name, '是否满足首选测试条件：', ok_)
                print()

        # print(final_data)
        # raise
    processed_count += 1
    if processed_count % 20 == 0:
        print(f"已处理 {processed_count} / {len(unique_papers)} 篇论文...")

print(f"\n完成！共处理 {processed_count} 篇论文。")
# -----------------------------
# 步骤4: 输出统计结果
# -----------------------------
print("\n" + "=" * 60)
# print(final_data)

all_props = []
for prop_name, keywords in property_keywords.items():
    all_props.append(prop_name)

output_rows = []
header = ['Molecule_ID']
output_rows.append(header)

for prop in all_props:
    header.extend([prop, f'condition {prop}'])

for mid, props in final_data.items():
    row = [mid]
    for prop in all_props:
        values = props.get(prop, [])
        if not values:
            row.extend(['', ''])  # 数值和条件都为空
        elif len(values) == 1:
            entry = values[0]
            row.extend([entry['val'], entry['condition']])
        else:
            # 多个值：计算均值，并找出最接近均值的那个条目
            vals = [v['val'] for v in values]
            mean_val = statistics.mean(vals)
            closest_entry = min(values, key=lambda x: abs(x['val'] - mean_val))
            row.extend([closest_entry['val'], closest_entry['condition']])
    output_rows.append(row)

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

print(f"✅ CSV 已保存为 {output_path}")



