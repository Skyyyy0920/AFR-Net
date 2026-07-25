import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import cm
import scipy.sparse.csgraph as csgraph

# ================= 配置区域 =================

# PPMI 数据路径 (pickle 文件)
DATA_PATH = "../data/PPMI/test_data.pkl"

# 输出图片路径
OUTPUT_FILE = "PPMI_Diamond_Matrix.png"

# 【关键选项】
PLOT_SHORTEST_PATH_INSTEAD_OF_FC = True

# 绘图参数
FIG_SIZE = (10, 10)
CMAP_SC = 'Reds'  # 左侧 SC 的颜色
CMAP_FC = 'Reds'  # 右侧 FC (或最短路径) 的颜色


# ===========================================

def load_ppmi_data(path):
    """加载并计算 PPMI 数据的平均矩阵"""
    print(f"正在加载数据: {path} ...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    scn = data['scn']
    fcn = data['fcn']

    print(f"计算平均矩阵 (样本数: {scn.shape[0]})...")
    avg_sc = np.mean(scn, axis=0)
    avg_fc = np.mean(fcn, axis=0)

    return avg_sc, avg_fc


def compute_shortest_path(adj_matrix):
    """基于邻接矩阵计算最短路径长度矩阵"""
    dist_matrix = np.zeros_like(adj_matrix)
    mask = adj_matrix > 0
    dist_matrix[mask] = 1.0 / (adj_matrix[mask])

    shortest_paths = csgraph.shortest_path(dist_matrix, method='auto', directed=False)
    shortest_paths[np.isinf(shortest_paths)] = 0
    return shortest_paths


def normalize_matrix(mat, log_scale=False):
    """归一化矩阵到 [0, 1]"""
    m = mat.copy()
    if log_scale:
        m = np.log1p(m)
    v_min, v_max = m.min(), m.max()
    if v_max > v_min:
        m = (m - v_min) / (v_max - v_min)
    return m


def plot_diamond_heatmap(sc_data, right_data, right_label="FC"):
    """
    绘制菱形热力图 - 简洁版本，只保留热力图和colorbar
    """
    N = sc_data.shape[0]

    # 归一化数据
    sc_norm = normalize_matrix(sc_data, log_scale=True)
    right_norm = normalize_matrix(right_data, log_scale=False)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 创建坐标网格
    x = np.arange(N + 1)
    y = np.arange(N + 1)
    X, Y = np.meshgrid(x, y)

    # 旋转45度
    transform = mtransforms.Affine2D().rotate_deg(45) + ax.transData

    # 绘制左半边 (下三角, SC)
    mask_sc = np.tri(N, k=-1, dtype=bool)
    sc_masked = np.ma.array(sc_norm, mask=~mask_sc)
    mesh_sc = ax.pcolormesh(X, Y, np.flipud(sc_masked), cmap=CMAP_SC,
                            transform=transform, shading='flat', vmin=0, vmax=1)

    # 绘制右半边 (上三角, FC/Path)
    mask_right = np.tri(N, k=0, dtype=bool).T
    right_masked = np.ma.array(right_norm, mask=~mask_right)
    mesh_right = ax.pcolormesh(X, Y, np.flipud(right_masked), cmap=CMAP_FC,
                               transform=transform, shading='flat', vmin=0, vmax=1)

    ax.set_aspect('equal')
    ax.axis('off')

    # 调整视野以包含整个菱形
    boundary = N * np.sqrt(2) / 2
    ax.set_xlim(-boundary - 5, boundary + 5)
    ax.set_ylim(-5, N * np.sqrt(2) + 5)

    # 添加小的colorbar到右下角
    # 创建一个inset axes用于colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # 在右下角创建一个小的colorbar
    axins = inset_axes(ax,
                       width="5%",  # colorbar宽度
                       height="80%",  # colorbar高度
                       loc='lower right',
                       bbox_to_anchor=(0.05, 0.05, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)

    cbar = plt.colorbar(mesh_sc, cax=axins)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"绘图完成，已保存至: {OUTPUT_FILE}")
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}")
        return

    avg_sc, avg_fc = load_ppmi_data(DATA_PATH)

    if PLOT_SHORTEST_PATH_INSTEAD_OF_FC:
        print("模式: 绘制最短路径 (Shortest Path)...")
        right_data = compute_shortest_path(avg_sc)
        right_label = "Shortest Paths\nSkeleton"
    else:
        print("模式: 绘制功能连接 (FC)...")
        right_data = np.abs(avg_fc)
        right_label = "Functional\nConnectivity"

    plot_diamond_heatmap(avg_sc, right_data, right_label=right_label)


if __name__ == "__main__":
    main()