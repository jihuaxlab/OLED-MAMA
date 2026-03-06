import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ==============================
# Step 1: 加载数据
# ==============================
df_props = pd.read_csv(r'D:\TADF_project\pdf_decode_chem_project\data\mol_PL_data_with_condition.csv')
# df_props = pd.read_csv(r'D:\工作报告\化学计算\数据\llm_vl\mol_PL_data_with_condition.csv')
df_ecfp_mol = pd.read_csv(r'D:\TADF_project\pdf_decode_chem_project\data\smiles_to_id_mapping_ecfp_12.csv')  # 分子的 ECFP
df_ecfp_cond = pd.read_csv(
    r'D:\TADF_project\pdf_decode_chem_project\data\mol_energy_data_condition_ecfp_fingerprints.csv')  # condition 的 ECFP

n_bits_mol = 2048

df_props.rename(columns={
    'λem': 'λem(nm)', 'Φ PLQY': 'PLQY(%)', 'condition Φ PLQY': 'condition PLQY',
}, inplace=True)

# ==============================
# Step 2: 构建 condition ECFP 的字典（name -> bit list）
# ==============================
# df_ecfp_cond 是展开格式：第一列是 'name'，后面是 ecfp_0, ecfp_1, ...
cond_ecfp_dict = {}
cond_state_dict = {}
for _, row in df_ecfp_cond.iterrows():
    cond_name = row['name']
    # 取所有 ecfp_* 列的值（按顺序）
    bits = [int(row[col]) for col in df_ecfp_cond.columns if col.startswith('ecfp_')]
    cond_ecfp_dict[cond_name] = bits
    cond_state_dict[cond_name] = row['State']

print(f"Loaded {len(cond_ecfp_dict)} condition ECFPs.")

# ==============================
# Step 3: 构建分子 ECFP 字典（Molecule_ID -> bit list）
# ==============================
mol_ecfp_dict = {}
for _, row in df_ecfp_mol.iterrows():
    mol_id = row['Molecule_ID']
    ecfp_str = str(row['ECFP_Bits']).strip()
    bits = [int(c) for c in ecfp_str]
    mol_ecfp_dict[mol_id] = bits

# ==============================
# Step 4: 展开 df_props 为长格式，并构建 X, y
# ==============================
records = []
X_list = []
y_list = []
mol_id_list = []
target_list = []

for _, row in df_props.iterrows():
    mol_id = row['Molecule_ID']

    # 跳过没有分子 ECFP 的
    if mol_id not in mol_ecfp_dict:
        continue

    mol_fp = mol_ecfp_dict[mol_id]

    # 处理 λem
    if pd.notna(row['λem(nm)']) and pd.notna(row['condition λem']):
        cond = row['condition λem']
        if cond in cond_ecfp_dict:
            cond_fp = cond_ecfp_dict[cond]
            state = [0]
            if cond_state_dict[cond] == 'solid':
                state = [1]
            fp_combined = mol_fp + cond_fp + state # 拼接 list
            X_list.append(fp_combined)
            y_list.append(float(row['λem(nm)']))  # 可以有负实验值
            mol_id_list.append(mol_id)
            target_list.append('λem(nm)')

    # 处理 PLQY
    if pd.notna(row['PLQY(%)']) and pd.notna(row['condition PLQY']):
        cond = row['condition PLQY']
        if cond in cond_ecfp_dict:
            cond_fp = cond_ecfp_dict[cond]
            state = [0]
            if cond_state_dict[cond] == 'solid':
                state = [1]
            fp_combined = mol_fp + cond_fp + state
            X_list.append(fp_combined)
            y_list.append(abs(float(row['PLQY(%)'])))
            mol_id_list.append(mol_id)
            target_list.append('PLQY(%)')

    # 处理 FWHM
    if pd.notna(row['FWHM']) and pd.notna(row['condition FWHM']):
        cond = row['condition FWHM']
        if cond in cond_ecfp_dict:
            cond_fp = cond_ecfp_dict[cond]
            state = [0]
            if cond_state_dict[cond] == 'solid':
                state = [1]
            fp_combined = mol_fp + cond_fp + state
            X_list.append(fp_combined)
            y_list.append(abs(float(row['FWHM'])))
            mol_id_list.append(mol_id)
            target_list.append('FWHM')

# 转为 numpy array
X = np.array(X_list, dtype=int)
y = np.array(y_list)
mol_ids_array = np.array(mol_id_list)
targets = np.array(target_list)

print(f"Total samples: {X.shape[0]}, Feature dim: {X.shape[1]}")

# ==============================
# Step 5: 按目标训练模型
# ==============================
plt.figure(figsize=(18, 5))
for i, target in enumerate(['λem(nm)', 'PLQY(%)', 'FWHM'], 1):
    mask = targets == target
    X_target = X[mask]
    y_target = y[mask]
    mol_ids_target = mol_ids_array[mask]

    if len(y_target) < 10:
        print(f"⚠️ {target} 样本太少 ({len(y_target)})，跳过")
        continue

    if target == 'PLQY(%)':
        valid = (y_target <= 100) & (y_target >= 0)
        X_target, y_target, mol_ids_target = X_target[valid], y_target[valid], mol_ids_target[valid]

    if target == 'FWHM':
        valid = (y_target <= 100) & (y_target >= 0)
        X_target, y_target, mol_ids_target = X_target[valid], y_target[valid], mol_ids_target[valid]

    if target == 'λem(nm)':
        valid = (y_target <= 1000) & (y_target >= 0)
        X_target, y_target, mol_ids_target = X_target[valid], y_target[valid], mol_ids_target[valid]

    X_train, X_test, y_train, y_test, mol_id_train, mol_id_test = train_test_split(
        X_target, y_target, mol_ids_target, test_size=0.20, random_state=42
    )
    print("DATA counts: ", X_target.shape[0], "Train set: ", X_train.shape, "   Test set:", X_test.shape)

    # --- 提取 without_cond 特征 ---
    # 直接相加没必要变为 0/1, 再将测试条件进行拼接
    # X_train_with = np.concatenate([X_train[:, :n_bits_mol] + X_train[:, n_bits_mol: 2 * n_bits_mol],
    #                                (X_train[:, -1]).reshape(-1, 1)], axis=1)
    # X_test_with = np.concatenate([X_test[:, :n_bits_mol] + X_test[:, n_bits_mol: 2 * n_bits_mol],
    #                               (X_test[:, -1]).reshape(-1, 1)], axis=1)

    X_train_with = X_train
    X_test_with = X_test

    # X_train_with = X_train[:, :n_bits_mol] + X_train[:, n_bits_mol: 2 * n_bits_mol]
    # X_test_with = X_test[:, :n_bits_mol] + X_test[:, n_bits_mol: 2 * n_bits_mol]

    X_train_without = X_train[:, :n_bits_mol]
    X_test_without = X_test[:, :n_bits_mol]
    # print(X_train[:, n_bits_mol:].shape, X_test[:, :n_bits_mol].shape)

    # ==============================
    # 模型 1: With Condition (Full)
    # ==============================
    model_with = XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                              subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0,
                              reg_lambda=1.0, gamma=0.5,  random_state=42)
    model_with.fit(X_train_with, y_train)
    y_pred_with = model_with.predict(X_test_with)
    r2_with = r2_score(y_test, y_pred_with)
    mae_with = mean_absolute_error(y_test, y_pred_with)

    outlier_mask = (y_pred_with - y_test) > 30
    print(f"发现 {outlier_mask.sum()} 个异常预测样本")
    outlier_mol_ids = mol_id_test[outlier_mask]
    outlier_true_vals = y_test[outlier_mask]
    outlier_pred_vals = y_pred_with[outlier_mask]

    for ii, mol_id in enumerate(outlier_mol_ids):
        print(f"Molecule_ID: {mol_id} | True {target}: {outlier_true_vals[ii]:.1f} | Pred {target}: {outlier_pred_vals[ii]:.1f}")

    # raise

    # ==============================
    # 模型 2: Without Condition (Mol Only)
    # ==============================
    model_without = XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                                 subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0,
                                 reg_lambda=1.0, gamma=0.5,  random_state=42)
    model_without.fit(X_train_without, y_train)
    y_pred_without = model_without.predict(X_test_without)
    r2_without = r2_score(y_test, y_pred_without)
    mae_without = mean_absolute_error(y_test, y_pred_without)

    # ==============================
    # Plot 1: With Condition (第1行)
    # ==============================
    plt.subplot(2, 3, i)  # 第1行，第i列
    plt.scatter(y_test, y_pred_with, alpha=0.7, color='steelblue', edgecolors='k', s=30)
    lims = [min(y_test.min(), y_pred_with.min()), max(y_test.max(), y_pred_with.max())]
    plt.plot(lims, lims, 'r--', lw=2)
    plt.title(f'{target} (With Condition)\nR²={r2_with:.3f}, MAE={mae_with:.2f}')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.grid(True, linestyle='--', alpha=0.5)

    # ==============================
    # Plot 2: Without Condition (第2行)
    # ==============================
    plt.subplot(2, 3, i + 3)  # 第2行，第i列
    plt.scatter(y_test, y_pred_without, alpha=0.7, color='orange', edgecolors='k', s=30)
    lims = [min(y_test.min(), y_pred_without.min()), max(y_test.max(), y_pred_without.max())]
    plt.plot(lims, lims, 'r--', lw=2)
    plt.title(f'{target} (Without Condition)\nR²={r2_without:.3f}, MAE={mae_without:.2f}')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('xgboost_comparison_2x3_with_vs_without_condition.png', dpi=300, bbox_inches='tight')
plt.show()
