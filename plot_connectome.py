import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from torch.utils.data import DataLoader

# 引入你的项目代码
from dial.model import DIALModel
from dial.data import load_data, preprocess_labels, ABCDDataset

# ================= 配置区域 =================
# 1. 路径设置
# 请根据需要取消注释对应的模型路径
# MODEL_PATH = "/data/tianhao/DIAL/results/OCD/20251202-184025/best_model.pth"
MODEL_PATH = "/data/tianhao/DIAL/results/Anx/20251202-184115/best_model.pth"
DATA_PATH = r"/data/tianhao/DIAL/data/data_dict_OCD.pkl"
COORD_FILE = "/data/tianhao/DIAL_plot/R/glasser.csv"
OUTPUT_DIR = "./plots_comparison"  # 图片保存目录

# 2. 参数设置
ARGS = {
    'task': 'Anx',  # 记得修改这里以匹配模型
    'd_model': 64,
    'num_node_layers': 2,
    'num_graph_layers': 2,
    'dropout': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

TOP_K = 200
NUM_SAMPLES_PER_CLASS = 5  # 每类画几个样本


# ===========================================

def read_csv_safe(path):
    """尝试多种编码读取 CSV 文件"""
    encodings = ['utf-8', 'gbk', 'ISO-8859-1', 'cp1252']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            try:
                return pd.read_csv(path, encoding=enc, skiprows=1)
            except:
                continue
    raise ValueError(f"无法读取文件 {path}")


def load_glasser_coords(node_names, csv_path):
    """读取坐标文件并对齐节点"""
    df = read_csv_safe(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 映射列名
    col_map = {}
    required_cols = ['ROI.x', 'x.mni', 'y.mni', 'z.mni']
    for req in required_cols:
        for col in df.columns:
            if req.lower() == col.lower():
                col_map[req] = col
                break

    coord_dict = {}
    # 填充字典
    rows_iter = df.iterrows()
    for _, row in rows_iter:
        try:
            if len(col_map) < 4:
                # 备用索引方案
                name = str(row.iloc[2]).strip()
                coords = [float(row.iloc[4]), float(row.iloc[5]), float(row.iloc[6])]
            else:
                name = str(row[col_map['ROI.x']]).strip()
                coords = [float(row[col_map['x.mni']]), float(row[col_map['y.mni']]), float(row[col_map['z.mni']])]
            coord_dict[name] = coords
        except:
            continue

    # 匹配坐标
    final_coords = []
    for name in node_names:
        clean_name = name.replace("_ROI", "").strip()
        if clean_name.startswith("sub-"): clean_name = clean_name.split("_", 1)[1]

        matched = False
        if clean_name in coord_dict:
            final_coords.append(coord_dict[clean_name])
            matched = True
        else:
            for key in coord_dict:
                if clean_name == key or clean_name in key:
                    final_coords.append(coord_dict[key])
                    matched = True
                    break
        if not matched:
            final_coords.append([0.0, 0.0, 0.0])

    return np.array(final_coords)


def get_adj_matrix_topk(matrix, k, use_abs=False):
    """
    从矩阵中提取 Top-K 边，返回二值化的邻接矩阵
    matrix: (N, N) numpy array
    k: int
    use_abs: bool, 是否按绝对值排序 (用于 FC)
    """
    N = matrix.shape[0]
    adj = np.zeros((N, N))

    # 忽略对角线和下三角
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)
    values = matrix[mask]

    if use_abs:
        sort_values = np.abs(values)
    else:
        sort_values = values

    # 获取 Top-K 的阈值
    if k >= len(values):
        threshold = -np.inf
    else:
        # partition 效率比 sort 高，找到第 k 大的值
        threshold = np.partition(sort_values, -k)[-k]

    # 构建邻接矩阵 (大于等于阈值的边置为 1)
    # 注意：这里可能会因为数值相同选出略多于 K 条边，但这对于绘图是可以接受的
    selected_mask = (sort_values >= threshold)

    # 还原到矩阵
    # 为了准确还原位置，我们重新遍历一下或者利用索引
    # 这里用一种简单的方法：获取 Top-K 索引
    flat_indices = np.argsort(sort_values)[::-1][:k]

    # 获取上三角的所有坐标
    rows, cols = np.triu_indices(N, k=1)

    # 选出 Top-K 的坐标
    sel_rows = rows[flat_indices]
    sel_cols = cols[flat_indices]

    adj[sel_rows, sel_cols] = 1
    adj[sel_cols, sel_rows] = 1  # 对称

    return adj


def plot_and_save(adj_matrix, coords, title, filename):
    """绘图辅助函数"""
    fig = plt.figure(figsize=(10, 8))

    display = plotting.plot_connectome(
        adjacency_matrix=adj_matrix,
        node_coords=coords,
        node_size=15,
        colorbar=False,
        node_color='auto',
        display_mode='lzry',
        # title=title,
        annotate=False,
        figure=None  # 让它创建自己的figure
    )

    # 获取 display 创建的 figure
    fig = display.frame_axes.figure

    fig.suptitle(title, fontsize=12, x=0.02, y=0.98,
                 verticalalignment='top', horizontalalignment='left', color='white',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='black', edgecolor='none'))

    # 手动添加一个紧凑的colorbar
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    from matplotlib import cm

    cbar_ax = fig.add_axes([0.85, 0.92, 0.1, 0.02])  # 顶部中央位置
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Reds')
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.ax.tick_params(labelsize=8)
    # 将刻度标签放在colorbar上方
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')
    # plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> 已保存: {os.path.basename(filename)}")


def main():
    device = ARGS['device']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== 多模态对比分析 | Task: {ARGS['task']} | Top-K: {TOP_K} ===")

    # 1. 加载数据
    print(f"[1/4] 加载数据: {DATA_PATH}")
    raw_data = load_data(DATA_PATH)
    processed_dict = preprocess_labels(raw_data, task=ARGS['task'])

    # 2. 筛选样本 (5个 HC, 5个 Patient)
    print(f"[2/4] 筛选样本 (每类 {NUM_SAMPLES_PER_CLASS} 个)...")
    samples_0 = []  # HC
    samples_1 = []  # Patient

    for key, item in processed_dict.items():
        if item['label'] == 0 and len(samples_0) < NUM_SAMPLES_PER_CLASS:
            samples_0.append(item)
        elif item['label'] == 1 and len(samples_1) < NUM_SAMPLES_PER_CLASS:
            samples_1.append(item)

        if len(samples_0) >= NUM_SAMPLES_PER_CLASS and len(samples_1) >= NUM_SAMPLES_PER_CLASS:
            break

    target_samples = samples_0 + samples_1
    print(f"  -> 选中样本数: {len(target_samples)} (0: {len(samples_0)}, 1: {len(samples_1)})")

    # 3. 准备模型与坐标
    print(f"[3/4] 初始化模型与坐标...")
    # 获取 ROI 名称 (用第一个样本)
    N = target_samples[0]['SC'].shape[0]
    df_temp = read_csv_safe(COORD_FILE)
    if 'ROI.x' in df_temp.columns:
        roi_names = df_temp['ROI.x'].dropna().astype(str).tolist()
    else:
        roi_names = df_temp.iloc[:, 2].dropna().astype(str).tolist()

    if len(roi_names) > N:
        roi_names = roi_names[:N]
    elif len(roi_names) < N:
        roi_names += [f"Node_{i}" for i in range(len(roi_names), N)]
    coords = load_glasser_coords(roi_names, COORD_FILE)

    model = DIALModel(
        N=N, d_model=ARGS['d_model'], num_classes=2, task='classification',
        num_node_layers=ARGS['num_node_layers'], num_graph_layers=ARGS['num_graph_layers'],
        dropout=ARGS['dropout']
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("  -> 模型权重已加载")
    else:
        print("  -> [警告] 权重未找到，使用随机初始化模型！")
    model.eval()

    # 4. 循环处理并绘图
    print(f"[4/4] 开始批量绘图 (输出目录: {OUTPUT_DIR})...")

    for i, sample in enumerate(target_samples):
        label_str = "HC" if sample['label'] == 0 else "Patient"
        sample_name = sample.get('name', f"sample_{i}")
        # 清理文件名中的非法字符
        safe_name = sample_name.replace("/", "_").replace("\\", "_")
        prefix = f"{i + 1:02d}_{label_str}_{safe_name}"

        print(f"\n处理样本 {i + 1}/{len(target_samples)}: {prefix}")

        # --- A. 绘制 SC (Structural Connectivity) ---
        sc_matrix = sample['SC']
        if isinstance(sc_matrix, torch.Tensor): sc_matrix = sc_matrix.numpy()
        adj_sc = get_adj_matrix_topk(sc_matrix, TOP_K, use_abs=False)
        plot_and_save(adj_sc, coords,
                      f"SC Top-{TOP_K}\n{label_str}: {sample_name}",
                      os.path.join(OUTPUT_DIR, f"{prefix}_SC.png"))

        # --- B. 绘制 FC (Functional Connectivity) ---
        # 注意：确认 key 是 'FC' 还是 'F'。ABCDDataset 常用 'FC' 或 'corr'
        # 根据之前代码 batch['F']，原始 dict 里通常也是 'FC'
        if 'FC' in sample:
            fc_matrix = sample['FC']
        elif 'F' in sample:
            fc_matrix = sample['F']
        else:
            print("  [跳过] 未找到 FC 数据")
            fc_matrix = None

        if fc_matrix is not None:
            if isinstance(fc_matrix, torch.Tensor): fc_matrix = fc_matrix.numpy()
            # FC 包含负值，按绝对值排序找最强连接
            adj_fc = get_adj_matrix_topk(fc_matrix, TOP_K, use_abs=True)
            plot_and_save(adj_fc, coords,
                          f"FC Top-{TOP_K}\n{label_str}: {sample_name}",
                          os.path.join(OUTPUT_DIR, f"{prefix}_FC.png"))

        # --- C. 绘制 Model Attention ---
        # 构造单样本 batch
        dataset = ABCDDataset([sample], device=device)
        loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate)
        batch = next(iter(loader))

        with torch.no_grad():
            _, edge_indices_list, m_list = model.inference(
                node_feat=batch['node_feat'].to(device),
                in_degree=batch['in_degree'].to(device),
                out_degree=batch['out_degree'].to(device),
                path_data=batch['path_data'].to(device),
                dist=batch['dist'].to(device),
                attn_mask=batch['attn_mask'].to(device),
                S=batch['S'].to(device),
                F=batch['F'].to(device)
            )

        # 提取模型权重
        model_weights = m_list[0].cpu()
        model_indices = edge_indices_list[0].cpu()

        # 构造模型邻接矩阵
        # 这里比较特殊，因为 edge_indices 是稀疏格式，我们需要根据 weights 选 Top-K
        k_actual = min(TOP_K, model_weights.shape[0])
        topk_vals, topk_inds = torch.topk(model_weights, k=k_actual)

        src_nodes = model_indices[0, topk_inds].numpy()
        dst_nodes = model_indices[1, topk_inds].numpy()

        adj_model = np.zeros((N, N))
        adj_model[src_nodes, dst_nodes] = 1
        adj_model[dst_nodes, src_nodes] = 1

        plot_and_save(adj_model, coords,
                      f"Information Load Top-{k_actual}\n{label_str}: {sample_name}",
                      os.path.join(OUTPUT_DIR, f"{prefix}_Model.png"))

    print("\n任务完成！所有图片已保存。")


if __name__ == "__main__":
    main()