import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import SpectralClustering
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from afr_net.data import preprocess_labels
except ImportError:
    preprocess_labels = None

# ================= 配置区域 =================
# DATA_PATH = r"../data/PPMI/test_data.pkl"
DATA_PATH = r"W:\Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing\data\ABCD\processed\data_dict.pkl"


TASK = 'Anx'
SAMPLE_IDX = 0
NUM_CLUSTERS = 6
OUTPUT_FILE = "sc_fc_composite_colorbar.png"


# ===========================================

def load_and_process_data(data_path, task_name):
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)

    if 'scn' in raw and 'fcn' in raw:
        print(f"[Info] Detected PPMI dataset format.")
        scn = raw['scn']
        fcn = raw['fcn']
        labels = raw.get('labels', np.zeros(len(scn)))
        samples = {}
        for i in range(len(scn)):
            key = f"PPMI_{i}"
            samples[key] = {
                'SC': scn[i],
                'FC': fcn[i],
                'label': int(labels[i]),
                'name': key
            }
        return samples
    else:
        print(f"[Info] Detected ABCD/Generic dictionary format.")
        if preprocess_labels is None:
            raise ImportError("Cannot import 'preprocess_labels'.")
        return preprocess_labels(raw, task=task_name)


def sort_indices_by_spectral(matrix, n_clusters=5):
    mat_abs = np.abs(matrix)
    mat_sym = (mat_abs + mat_abs.T) / 2
    try:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(mat_sym)
    except Exception as e:
        print(f"Clustering failed: {e}. Fallback to raw indices.")
        return np.arange(matrix.shape[0]), []

    sorted_idx = np.argsort(labels)
    sorted_labels = labels[sorted_idx]

    boundaries = []
    if len(sorted_labels) > 0:
        current_label = sorted_labels[0]
        start_idx = 0
        for i, label in enumerate(sorted_labels):
            if label != current_label:
                boundaries.append((start_idx, i - start_idx))
                current_label = label
                start_idx = i
        boundaries.append((start_idx, len(sorted_labels) - start_idx))

    return sorted_idx, boundaries


def plot_composite_matrix(ax, sc, fc, indices, boundaries, title):
    """
    绘制复合矩阵并返回 image 对象以便画 colorbar
    """
    sc_sorted = sc[indices][:, indices]
    fc_sorted = fc[indices][:, indices]
    n = sc_sorted.shape[0]

    mask_lower = np.tril(np.ones((n, n)), k=0).astype(bool)
    mask_upper = np.triu(np.ones((n, n)), k=1).astype(bool)

    # SC Plotting
    sc_log = np.log1p(sc_sorted)
    if sc_log.max() > 0:
        sc_plot = sc_log / sc_log.max()
    else:
        sc_plot = sc_log
    sc_masked = np.ma.masked_array(sc_plot, mask=~mask_lower)
    im_sc = ax.imshow(sc_masked, cmap='Greens', interpolation='nearest', aspect='equal', vmin=0, vmax=1)

    # FC Plotting
    fc_masked = np.ma.masked_array(fc_sorted, mask=~mask_upper)
    im_fc = ax.imshow(fc_masked, cmap='RdBu_r', interpolation='nearest', aspect='equal', vmin=-1, vmax=1)

    # Boundaries
    for start, length in boundaries:
        rect = patches.Rectangle(
            (start - 0.5, start - 0.5), length, length,
            linewidth=2.0, edgecolor='red', facecolor='none', zorder=10
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=14, pad=10)
    ax.axis('off')

    return im_sc, im_fc


def main():
    processed_dict = load_and_process_data(DATA_PATH, TASK)
    if len(processed_dict) == 0: return

    sample = list(processed_dict.values())[SAMPLE_IDX]
    sc, fc = sample['SC'], sample['FC']
    if isinstance(sc, torch.Tensor): sc = sc.numpy()
    if isinstance(fc, torch.Tensor): fc = fc.numpy()

    print("Computing communities...")
    idx_sc, bounds_sc = sort_indices_by_spectral(sc, n_clusters=NUM_CLUSTERS)
    fc_abs = np.abs(fc)
    idx_fc, bounds_fc = sort_indices_by_spectral(fc_abs, n_clusters=NUM_CLUSTERS)

    # 调整画布大小，留出右侧空间给 colorbar
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(right=0.85, wspace=0.1)

    # Plot 1
    im_sc1, im_fc1 = plot_composite_matrix(
        axes[0], sc, fc, idx_sc, bounds_sc,
        title=""
    )
    # Plot 2
    im_sc2, im_fc2 = plot_composite_matrix(
        axes[1], sc, fc, idx_fc, bounds_fc,
        title=""
    )

    # 添加 Colorbars
    # 位置参数: [left, bottom, width, height]
    # 我们在图的最右侧添加两个竖条

    # 1. Structure Colorbar (Reds)
    cbar_ax_sc = fig.add_axes([0.88, 0.15, 0.02, 0.3])  # 下半部分
    cb_sc = fig.colorbar(im_sc1, cax=cbar_ax_sc)
    cb_sc.set_label('Structural Connection (Normalized)', rotation=270, labelpad=15)

    # 2. Function Colorbar (RdBu_r)
    cbar_ax_fc = fig.add_axes([0.88, 0.55, 0.02, 0.3])  # 上半部分
    cb_fc = fig.colorbar(im_fc1, cax=cbar_ax_fc)
    cb_fc.set_label('Functional Correlation (Pearson)', rotation=270, labelpad=15)

    # 添加 SC/FC 文本标签到图中角落，辅助说明
    for ax in axes:
        n = sc.shape[0]
        ax.text(0, n, "SC (Lower)", fontsize=10, ha='left', va='top', color='darkgreen', fontweight='bold')
        ax.text(n, 0, "FC (Upper)", fontsize=10, ha='right', va='bottom', color='darkred', fontweight='bold')

    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.tight_layout()
    print(f"Figure with colorbars saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()