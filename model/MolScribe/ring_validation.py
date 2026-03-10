import json
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist, squareform
from molscribe.chemistry import convert_graph_to_smiles
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os

BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]
BOND_TYPE_TO_ID = {name: idx for idx, name in enumerate(BOND_TYPES)}

# Key mapping
BOND_ORDER = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 1, 6: 1}

VALENCE_LIMIT = {'C': 4, 'N': 3, 'O': 2, 'B': 3, '[N+]': 4, '[O+]': 3, '[B-]': 4}



def load_json_data(data_path, trim_ratio=0.1, min_bond_count_for_trim=5, merge_strategy='max'):
    """
    加载并预处理分子结构，返回：
        atoms_element_list: list[str]
        coords: np.ndarray (M, 2)
        conf_adj: np.ndarray (M, M) —— 置信度邻接矩阵
        bond_type_adj: np.ndarray (M, M) —— 键类型ID矩阵 [0~6]
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    atoms = data["atom_info"]
    N = len(atoms)
    print(f'加载原子数量{N}')
    if N == 0:
        return [], np.zeros((0, 2)), np.zeros((0, 0)), np.zeros((0, 0))

    coords = np.zeros((N, 2), dtype=np.float32)
    atoms_element_list = []
    atoms_confidence_list = []
    for i, atom in enumerate(atoms):
        atoms_element_list.append(atom['atom_symbol'])
        atoms_confidence_list.append(atom['confidence'])
        coords[i, 0] = atom["x"]
        coords[i, 1] = atom["y"]

    # 初始化原始矩阵
    conf_adj = np.zeros((N, N), dtype=np.float32)
    bond_type_adj = np.zeros((N, N), dtype=np.int32)  # 键类型ID

    bonds = data.get("bond_info", [])
    for bond in bonds:
        u, v = bond["endpoint_atoms"]
        conf = bond["confidence"]
        bond_type_str = bond.get("bond_type", "single")  # 默认 single
        bond_type_id = BOND_TYPE_TO_ID.get(bond_type_str, 1)  # 未知类型默认为 single
        if 0 <= u < N and 0 <= v < N:
            conf_adj[u, v] = conf
            conf_adj[v, u] = conf
            bond_type_adj[u, v] = bond_type_id
            bond_type_adj[v, u] = bond_type_id

    # ===== Step 1: 计算平均键长 =====
    bond_lengths = []
    for i in range(N):
        for j in range(i + 1, N):
            if conf_adj[i, j] > 0:
                d = np.linalg.norm(coords[i] - coords[j])
                bond_lengths.append(d)

    if bond_lengths:
        bond_lengths = np.array(bond_lengths)
        if len(bond_lengths) >= min_bond_count_for_trim:
            n_trim = int(len(bond_lengths) * trim_ratio)
            if n_trim > 0:
                sorted_lengths = np.sort(bond_lengths)
                trimmed = sorted_lengths[n_trim:-n_trim] if n_trim * 2 < len(bond_lengths) else sorted_lengths
                avg_bond_length = np.mean(trimmed)
            else:
                avg_bond_length = np.mean(bond_lengths)
        else:
            avg_bond_length = np.mean(bond_lengths)
    else:
        all_dists = pdist(coords)
        all_dists = all_dists[all_dists > 1e-6]
        avg_bond_length = np.median(all_dists) * 2 if len(all_dists) > 0 else 1.0

    threshold = 0.1 * avg_bond_length

    # ===== Step 2: 原子合并（距离过近）=====
    parent = list(range(N))
    dist_matrix = squareform(pdist(coords))

    for i in range(N):
        for j in range(i + 1, N):
            if dist_matrix[i, j] < threshold:
                # 找根
                def find_root(x):
                    while parent[x] != x:
                        x = parent[x]
                    return x
                root_i, root_j = find_root(i), find_root(j)
                if root_i == root_j:
                    continue

                # 保留高置信度
                if atoms_confidence_list[root_i] >= atoms_confidence_list[root_j]:
                    keep, remove = root_i, root_j
                else:
                    keep, remove = root_j, root_i

                # 转移 remove 的所有键到 keep
                for k in range(N):
                    if k == keep or k == remove:
                        continue
                    if conf_adj[remove, k] > 0:
                        bt = bond_type_adj[remove, k]
                        conf_val = conf_adj[remove, k]

                        if conf_adj[keep, k] > 0:
                            # 已存在键：合并策略
                            if merge_strategy == 'max':
                                if conf_val > conf_adj[keep, k]:
                                    conf_adj[keep, k] = conf_val
                                    conf_adj[k, keep] = conf_val
                                    bond_type_adj[keep, k] = bt
                                    bond_type_adj[k, keep] = bt
                                # 否则保留原键
                            elif merge_strategy == 'mean':
                                conf_adj[keep, k] = (conf_adj[keep, k] + conf_val) / 2
                                conf_adj[k, keep] = conf_adj[keep, k]
                                # 键类型保留原或新？这里保守保留原（或可选更高置信度的类型）
                        else:
                            # 无冲突，直接转移
                            conf_adj[keep, k] = conf_val
                            conf_adj[k, keep] = conf_val
                            bond_type_adj[keep, k] = bt
                            bond_type_adj[k, keep] = bt

                # 清除 remove 的连接
                conf_adj[remove, :] = 0
                conf_adj[:, remove] = 0
                bond_type_adj[remove, :] = 0
                bond_type_adj[:, remove] = 0
                parent[remove] = keep

    # ===== Step 3: 构建新索引 =====
    final_indices = [i for i in range(N) if parent[i] == i]
    if not final_indices:
        M = 0
        return [], np.zeros((0, 2)), np.zeros((0, 0)), np.zeros((0, 0))

    old_to_new = {old: new for new, old in enumerate(final_indices)}
    M = len(final_indices)

    new_elements = [atoms_element_list[i] for i in final_indices]
    print("new_elements: ",new_elements)
    new_coords = coords[final_indices]
    new_conf_adj = np.zeros((M, M), dtype=np.float32)
    new_bond_type_adj = np.zeros((M, M), dtype=np.int32)

    for i_old in final_indices:
        for j_old in final_indices:
            if i_old == j_old:
                continue
            if conf_adj[i_old, j_old] > 0:
                i_new = old_to_new[i_old]
                j_new = old_to_new[j_old]
                new_conf_adj[i_new, j_new] = conf_adj[i_old, j_old]
                new_bond_type_adj[i_new, j_new] = bond_type_adj[i_old, j_old]

    # ===== Step 4: 价态校正（断开超价单键）=====
    for idx, elem in enumerate(new_elements):
        elem_upper = elem.upper()
        max_valence = VALENCE_LIMIT.get(elem_upper, None)

        if max_valence is None:
            continue  # 不限制

        # 特殊规则：根据邻接原子类型动态提升最大价态
        if elem_upper == 'B':
            # 硼可形成四配位（如 B-O, B-F, B-N），此时价态上限为 4
            has_tetrahedral_ligand = False
            for j in range(M):
                if new_bond_type_adj[idx, j] > 0:
                    neighbor_elem = new_elements[j].upper()
                    if neighbor_elem in {'O', 'F', 'N'}:
                        has_tetrahedral_ligand = True
                        break
            if has_tetrahedral_ligand:
                max_valence = 4  # 允许四配位硼

        elif elem_upper == 'O':
        # 氧与硼成键时可带正电或形成配位键（如 B←O），此时可视为三价
            has_boron_neighbor = any(
                new_bond_type_adj[idx, j] > 0 and new_elements[j].upper() == 'B'
                for j in range(M)
            )
            if has_boron_neighbor:
                max_valence = 3

        elif elem_upper == 'N':
            # 氮与硼成键可能参与配位（如胺-硼烷 adduct），形成四配位氮
            has_boron_neighbor = any(
                new_bond_type_adj[idx, j] > 0 and new_elements[j].upper() == 'B'
                for j in range(M)
            )
            if has_boron_neighbor:
                max_valence = 4
            # print(elem_upper, " max ", max_valence)

        # 计算当前总键级
        total_order = 0
        single_bond_neighbors = []  # (neighbor_idx, confidence)
        for j in range(M):
            if j == idx:
                continue
            bt = new_bond_type_adj[idx, j]
            if bt == 0:
                continue
            order = BOND_ORDER[bt]
            total_order += order
            if bt == 1 or bt == 2:  # 单/双 键
                single_bond_neighbors.append((j, new_conf_adj[idx, j]))

        # 如果超价，删除置信度低的键
        while total_order > max_valence and single_bond_neighbors:
            # 按置信度升序（最低的先删）
            single_bond_neighbors.sort(key=lambda x: x[1])
            j_remove, _ = single_bond_neighbors.pop(0)

            # 断开键
            bt_removed = new_bond_type_adj[idx, j_remove]
            order_removed = BOND_ORDER[bt_removed]

            new_conf_adj[idx, j_remove] = 0
            new_conf_adj[j_remove, idx] = 0
            new_bond_type_adj[idx, j_remove] = 0
            new_bond_type_adj[j_remove, idx] = 0

            total_order -= order_removed

            # 更新剩余单键列表（重新计算更安全，但这里简单移除）
            single_bond_neighbors = [(j, conf) for j, conf in single_bond_neighbors if j != j_remove]
    print(f'清洗后原子数量{len(new_elements)}')
    return new_elements, new_coords, new_conf_adj, new_bond_type_adj

def find_all_simple_cycles(adj_matrix):
    """
    找出无向图中所有的简单环（simple cycles）。
    
    参数:
        adj_matrix: n x n 的邻接矩阵（对称，无向图）
    
    返回:
        List[List[int]]: 每个元素是一个环的节点列表（如 [0,1,2,0]）
    """
    n = len(adj_matrix)
    all_cycles = set()  # 用 set 去重（避免同一环不同起点/方向重复）

    def dfs(start, current, visited, path):
        """
        从 start 出发，当前在 current，已访问 visited，当前路径 path
        """
        for neighbor in range(n):
            if adj_matrix[current][neighbor] == 0:
                continue

            if neighbor == start and len(path) > 2:
                # 找到一个环：回到起点，且长度 >= 3
                cycle = tuple(path)
                # 标准化环表示：最小节点开头，且方向固定（取字典序小的方向）
                min_idx = cycle.index(min(cycle))
                rotated = cycle[min_idx:] + cycle[:min_idx]
                reversed_rotated = tuple(reversed(rotated))
                canonical = min(rotated, reversed_rotated)
                all_cycles.add(canonical)

            elif neighbor not in visited and neighbor > start:
                # 关键剪枝：只允许访问编号 > start 的节点，防止重复环
                # 这保证每个环只从其最小编号节点开始搜索
                dfs(start, neighbor, visited | {neighbor}, path + [neighbor])

    # 从每个节点作为“最小节点”开始搜索
    for i in range(n):
        dfs(i, i, {i}, [i])

    # 转换回带闭合的环（可选：添加起点结尾）
    result = []
    for cycle in all_cycles:
        result.append(list(cycle) + [cycle[0]])
    
    return result

def find_all_chordless_cycles(adj_matrix):
    """
    找出无向图中所有无弦简单环（chordless cycles / induced cycles）。
    
    参数:
        adj_matrix: n x n 的邻接矩阵（对称，无向图）
    
    返回:
        List[List[int]]: 每个元素是一个无弦环的节点列表（如 [0,1,2,0]）
    """
    n = len(adj_matrix)
    all_candidate_cycles = set()

    def dfs(start, current, visited, path):
        for neighbor in range(n):
            if adj_matrix[current][neighbor] == 0:
                continue

            if neighbor == start and len(path) > 2:
                # 找到一个简单环
                cycle = tuple(path)
                min_idx = cycle.index(min(cycle))
                rotated = cycle[min_idx:] + cycle[:min_idx]
                reversed_rotated = tuple(reversed(rotated))
                canonical = min(rotated, reversed_rotated)
                all_candidate_cycles.add(canonical)

            elif neighbor not in visited and neighbor > start:
                dfs(start, neighbor, visited | {neighbor}, path + [neighbor])

    # 生成所有简单环（作为候选）
    for i in range(n):
        dfs(i, i, {i}, [i])

    # 过滤出 chordless cycles
    chordless_cycles = []
    for cycle in all_candidate_cycles:
        cycle_len = len(cycle)
        if cycle_len < 3:
            continue

        # 构建环中节点的索引映射（用于判断是否相邻）
        node_to_index = {node: idx for idx, node in enumerate(cycle)}
        is_chordless = True

        # 检查每一对非相邻节点是否有边（即是否存在 chord）
        for i in range(cycle_len):
            for j in range(i + 2, cycle_len):
                # 跳过相邻节点（包括首尾相连的情况）
                if i == 0 and j == cycle_len - 1:
                    continue  # 首尾在环中是相邻的

                u, v = cycle[i], cycle[j]
                if adj_matrix[u][v] != 0:
                    # 发现弦！
                    is_chordless = False
                    break
            if not is_chordless:
                break

        if is_chordless:
            chordless_cycles.append(list(cycle) + [cycle[0]])  # 闭合环

    return chordless_cycles

def visualize_molecule_with_rings(element_list, coords, adj_matrix, rings, save_path="molecule_with_rings.png"):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    n = len(coords)

    # --- 绘制化学键（所有边）---
    lines = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                lines.append([coords[i], coords[j]])
    if lines:
        lc = LineCollection(lines, colors='gray', linewidths=1.5, alpha=0.7)
        ax.add_collection(lc)

    # --- 绘制原子 ---
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c='black', s=2, zorder=5)

    # --- 高亮每个环 ---
    from itertools import cycle
    color_cycle = cycle(['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive'])
    
    for ring_nodes in rings:
        ring_coords = coords[ring_nodes]
        # 闭合多边形
        polygon = np.vstack([ring_coords, ring_coords[0]])
        ring_color = next(color_cycle)
        ax.plot(polygon[:, 0], polygon[:, 1], color=ring_color, linewidth=3, alpha=0.6)

    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=20, color='r', fontweight='bold')

    ax.set_aspect('equal')
    ax.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

def main(json_path, output_img_dir):
    base_name = os.path.basename(json_path)
    
    element_list, atoms_coords, confidence_matrix, bond_type_adj = load_json_data(json_path)
    
    smiles_list, molblock_list, r_success = convert_graph_to_smiles([atoms_coords], [element_list], [bond_type_adj], images=None, num_workers=1)
    print('转化为Smiles: ', r_success, smiles_list)
    all_mol_detected =  smiles_list[0].split('.')
    longest_smiles = max(all_mol_detected, key=len)
    
    print("所有片段:", all_mol_detected)
    print("最长的 SMILES:", longest_smiles)
    print("长度:", len(longest_smiles))

    is_valid = False
    output_img_path = ""
    if longest_smiles:
        mol = Chem.MolFromSmiles(longest_smiles)
    else:
        raise Exception('无效分子')

    if mol is None:
        print("❌ 无效的 SMILES")
    else:
        print('有效分子')
        is_valid = True
        AllChem.Compute2DCoords(mol)
        
        img = Draw.MolToImage(mol, size=(400, 400), kekulize=True)
        output_img_path = os.path.join(output_img_dir, f"{base_name.split('.')[0]}-recognition.png")
        img.save(output_img_path)
        print(f"✅ 分子图像已保存为 {output_img_path}")

    # Save SMILES result JSON
    smiles_json_path = os.path.join(output_img_dir, f"{base_name.split('.')[0]}-SMILES.json")
    with open(smiles_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "smiles": longest_smiles,
            "is_valid": is_valid,
            "img_path": output_img_path 
        }, f, indent=2)
    print(f"✅ SMILES 结果已保存至 {smiles_json_path}")

    adj_mx = (confidence_matrix > 0).astype(int)
    rings = find_all_chordless_cycles(adj_mx)
    print(rings)
    visualize_molecule_with_rings(
            element_list=element_list,
            coords=atoms_coords,
            adj_matrix=adj_mx,
            rings=rings,
            save_path=os.path.join(output_img_dir, f"{base_name.split('.')[0]}-rings.png")
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Process JSON data of molecules and generate images.")
    parser.add_argument('--json_path', type=str, default=None, help='Path to the input JSON file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated files. Defaults to the directory of json_path.')
    
    args = parser.parse_args()
    if args.json_path is None or not os.path.isfile(args.json_path):
        raise Exception("Invalid path to the input JSON file")
    json_path = os.path.abspath(args.json_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None

    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    print(output_dir)
    
    main(args.json_path, output_dir)


