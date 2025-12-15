import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from torch.utils.data import DataLoader

# Import your project code
from dial.model import DIALModel
from dial.data import load_data, preprocess_labels, ABCDDataset

# ================= Configuration =================
# 1. Path Settings
# Uncomment the corresponding model path as needed
# MODEL_PATH = "/data/tianhao/DIAL/results/OCD/20251202-184025/best_model.pth"
# MODEL_PATH = "/data/tianhao/DIAL/results/Anx/20251202-184115/best_model.pth"
MODEL_PATH = "/data/tianhao/DIAL/results/Bip/20251202-184110/best_model.pth"
DATA_PATH = r"/data/tianhao/DIAL/data/data_dict_OCD.pkl"
COORD_FILE = "/data/tianhao/DIAL_plot/R/glasser.csv"
OUTPUT_DIR = "./plots"

# 2. Parameter Settings
ARGS = {
    'task': 'Bip',  # Remember to modify this to match the model
    'd_model': 64,
    'num_node_layers': 2,
    'num_graph_layers': 2,
    'dropout': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

TOP_K = 200
NUM_SAMPLES_PER_CLASS = 5  # Number of samples per class to plot


# ===========================================

def read_csv_safe(path):
    """Try multiple encodings to read CSV files"""
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
    raise ValueError(f"Unable to read file {path}")


def load_glasser_coords(node_names, csv_path):
    """Read coordinate file and align nodes"""
    df = read_csv_safe(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Map column names
    col_map = {}
    required_cols = ['ROI.x', 'x.mni', 'y.mni', 'z.mni']
    for req in required_cols:
        for col in df.columns:
            if req.lower() == col.lower():
                col_map[req] = col
                break

    coord_dict = {}
    # Populate dictionary
    rows_iter = df.iterrows()
    for _, row in rows_iter:
        try:
            if len(col_map) < 4:
                # Fallback indexing scheme
                name = str(row.iloc[2]).strip()
                coords = [float(row.iloc[4]), float(row.iloc[5]), float(row.iloc[6])]
            else:
                name = str(row[col_map['ROI.x']]).strip()
                coords = [float(row[col_map['x.mni']]), float(row[col_map['y.mni']]), float(row[col_map['z.mni']])]
            coord_dict[name] = coords
        except:
            continue

    # Match coordinates
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
    Extract Top-K edges from matrix, return binarized adjacency matrix
    matrix: (N, N) numpy array
    k: int
    use_abs: bool, whether to sort by absolute value (for FC)
    """
    N = matrix.shape[0]
    adj = np.zeros((N, N))

    # Ignore diagonal and lower triangle
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)
    values = matrix[mask]

    if use_abs:
        sort_values = np.abs(values)
    else:
        sort_values = values

    # Get Top-K threshold
    if k >= len(values):
        threshold = -np.inf
    else:
        # partition is more efficient than sort, find the k-th largest value
        threshold = np.partition(sort_values, -k)[-k]

    # Build adjacency matrix (edges >= threshold set to 1)
    # Note: Might select slightly more than K edges due to duplicate values, but acceptable for plotting
    selected_mask = (sort_values >= threshold)

    # Restore to matrix
    # To accurately restore positions, iterate again or use indices
    # Use a simple method here: Get Top-K indices
    flat_indices = np.argsort(sort_values)[::-1][:k]

    # Get all coordinates of the upper triangle
    rows, cols = np.triu_indices(N, k=1)

    # Select Top-K coordinates
    sel_rows = rows[flat_indices]
    sel_cols = cols[flat_indices]

    adj[sel_rows, sel_cols] = 1
    adj[sel_cols, sel_rows] = 1  # Symmetric

    return adj


def plot_and_save(adj_matrix, coords, title, filename):
    """Plotting helper function"""
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
        figure=None  # Let it create its own figure
    )

    # Get the figure created by display
    fig = display.frame_axes.figure

    fig.suptitle(f"{ARGS['task']}_{title}", fontsize=12, x=0.02, y=0.98,
                 verticalalignment='top', horizontalalignment='left', color='white',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='black', edgecolor='none'))

    # Manually add a compact colorbar
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    from matplotlib import cm

    cbar_ax = fig.add_axes([0.85, 0.92, 0.1, 0.02])  # Top center position
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Reds')
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.ax.tick_params(labelsize=8)
    # Place tick labels above colorbar
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')
    # plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {os.path.basename(filename)}")


def main():
    device = ARGS['device']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== Multimodal Comparative Analysis | Task: {ARGS['task']} | Top-K: {TOP_K} ===")

    # 1. Load Data
    print(f"[1/4] Loading data: {DATA_PATH}")
    raw_data = load_data(DATA_PATH)
    processed_dict = preprocess_labels(raw_data, task=ARGS['task'])

    # 2. Filter Samples (5 HC, 5 Patient)
    print(f"[2/4] Filtering samples ({NUM_SAMPLES_PER_CLASS} per class)...")
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
    print(f"  -> Selected samples: {len(target_samples)} (0: {len(samples_0)}, 1: {len(samples_1)})")

    # 3. Prepare Model and Coordinates
    print(f"[3/4] Initializing model and coordinates...")
    # Get ROI names (use the first sample)
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
        print("  -> Model weights loaded")
    else:
        print("  -> [Warning] Weights not found, using randomly initialized model!")
    model.eval()

    # 4. Loop processing and plotting
    print(f"[4/4] Starting batch plotting (Output dir: {OUTPUT_DIR})...")

    for i, sample in enumerate(target_samples):
        label_str = "HC" if sample['label'] == 0 else "Patient"
        sample_name = sample.get('name', f"sample_{i}")
        # Clean illegal characters in filenames
        safe_name = sample_name.replace("/", "_").replace("\\", "_")
        prefix = f"{ARGS['task']}_{i + 1:02d}_{label_str}_{safe_name}"

        print(f"\nProcessing sample {i + 1}/{len(target_samples)}: {prefix}")

        # --- A. Plot SC (Structural Connectivity) ---
        sc_matrix = sample['SC']
        if isinstance(sc_matrix, torch.Tensor): sc_matrix = sc_matrix.numpy()
        adj_sc = get_adj_matrix_topk(sc_matrix, TOP_K, use_abs=False)
        plot_and_save(adj_sc, coords,
                      f"SC Top-{TOP_K}\n{label_str}: {sample_name}",
                      os.path.join(OUTPUT_DIR, f"{prefix}_SC.png"))

        # --- B. Plot FC (Functional Connectivity) ---
        # Note: Confirm if key is 'FC' or 'F'. ABCDDataset often uses 'FC' or 'corr'
        # Based on previous code batch['F'], usually 'FC' in raw dict
        if 'FC' in sample:
            fc_matrix = sample['FC']
        elif 'F' in sample:
            fc_matrix = sample['F']
        else:
            print("  [Skip] FC data not found")
            fc_matrix = None

        if fc_matrix is not None:
            if isinstance(fc_matrix, torch.Tensor): fc_matrix = fc_matrix.numpy()
            # FC contains negative values, sort by absolute value to find strongest connections
            adj_fc = get_adj_matrix_topk(fc_matrix, TOP_K, use_abs=True)
            plot_and_save(adj_fc, coords,
                          f"FC Top-{TOP_K}\n{label_str}: {sample_name}",
                          os.path.join(OUTPUT_DIR, f"{prefix}_FC.png"))

        # --- C. Plot Model Attention ---
        # Construct single-sample batch
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

        # Extract model weights
        model_weights = m_list[0].cpu()
        model_indices = edge_indices_list[0].cpu()

        # Construct model adjacency matrix
        # This is special because edge_indices is sparse, need to select Top-K based on weights
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

    print("\nTask completed! All plots saved.")


if __name__ == "__main__":
    main()