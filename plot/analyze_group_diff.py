import os
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from nilearn import plotting
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your project code
from dial.model import DIALModel
from dial.data import load_data, preprocess_labels, ABCDDataset

# ================= Configuration (Please Verify) =================
# 1. Random Seed (Must match the seed in training code for reproducibility)
SEED = 20010920

# 2. Path Settings
# MODEL_PATH = "/data/tianhao/DIAL/results/OCD/20251202-184025/best_model.pth"
MODEL_PATH = "/data/tianhao/DIAL/results/Anx/20251202-184115/best_model.pth"
# MODEL_PATH = "/data/tianhao/DIAL/results/Bip/20251202-184110/best_model.pth"
DATA_PATH = r"/data/tianhao/DIAL/data/data_dict_OCD.pkl"
COORD_FILE = "/data/tianhao/DIAL_plot/R/glasser.csv"
OUTPUT_DIR = "./plots_group_diff"

# 3. Model Arguments (Must match model training configuration)
ARGS = {
    'task': 'Anx',
    'd_model': 64,
    'num_node_layers': 2,
    'num_graph_layers': 2,
    'dropout': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

TOP_K = 200


# =======================================================

def set_seed(seed):
    """Set global random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"-> Seed set to {seed}")


def read_csv_safe(path):
    """Read CSV safely"""
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


def load_glasser_coords(csv_path, num_nodes_expected):
    """Read coordinate file and return (N, 3) coordinate array"""
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
    for _, row in df.iterrows():
        try:
            if len(col_map) < 4:  # Fallback
                name = str(row.iloc[2]).strip()
                coords = [float(row.iloc[4]), float(row.iloc[5]), float(row.iloc[6])]
            else:
                name = str(row[col_map['ROI.x']]).strip()
                coords = [float(row[col_map['x.mni']]), float(row[col_map['y.mni']]), float(row[col_map['z.mni']])]
            coord_dict[name] = coords
        except:
            continue

    # Get name list (pad if mismatch)
    # Simplified logic: Read in CSV order, pad with 0 if insufficient
    # Better approach: Pass ROI names list here if available.
    # Assuming CSV order matches matrix order as specific ROI name order is unknown

    final_coords = []
    # Attempt to extract in order
    keys = list(coord_dict.keys())
    for i in range(num_nodes_expected):
        if i < len(keys):
            final_coords.append(coord_dict[keys[i]])
        else:
            final_coords.append([0.0, 0.0, 0.0])  # Padding

    return np.array(final_coords)


def get_balanced_data(processed_dict):
    """Reproduce 1:1 data balancing logic"""
    samples_0 = []  # HC
    samples_1 = []  # Patient

    # Sort to ensure determinism
    sorted_items = sorted(processed_dict.values(), key=lambda x: str(x.get('name', x.get('id', str(x)))))

    for item in sorted_items:
        if item['label'] == 0:
            samples_0.append(item)
        elif item['label'] == 1:
            samples_1.append(item)

    n_samples = min(len(samples_0), len(samples_1))
    print(f"-> Balancing to 1:1 ratio (N={n_samples} per class)...")

    np.random.shuffle(samples_0)
    np.random.shuffle(samples_1)

    return samples_0[:n_samples], samples_1[:n_samples]


def get_averaged_matrices(samples, model, device, num_nodes):
    """Calculate group average matrices"""
    accum_sc = np.zeros((num_nodes, num_nodes))
    accum_fc = np.zeros((num_nodes, num_nodes))
    accum_model = np.zeros((num_nodes, num_nodes))

    count = 0
    dataset = ABCDDataset(samples, device=device)
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            count += 1
            raw_sample = samples[count - 1]

            # SC
            sc = raw_sample['SC']
            if isinstance(sc, torch.Tensor): sc = sc.numpy()
            accum_sc += sc

            # FC
            if 'FC' in raw_sample:
                fc = raw_sample['FC']
            elif 'F' in raw_sample:
                fc = raw_sample['F']
            else:
                fc = np.zeros_like(sc)
            if isinstance(fc, torch.Tensor): fc = fc.numpy()
            accum_fc += np.abs(fc)

            # Model Attention
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
            weights = m_list[0].cpu().numpy()
            indices = edge_indices_list[0].cpu().numpy()
            temp = np.zeros((num_nodes, num_nodes))
            temp[indices[0], indices[1]] = weights
            accum_model += temp

    if count == 0: return None, None, None
    return accum_sc / count, accum_fc / count, accum_model / count


def get_topk_edge_set(matrix, k):
    """
    Return Top-K edge set {(u,v), ...}, where u < v
    """
    N = matrix.shape[0]
    sym_matrix = (matrix + matrix.T) / 2
    triu_indices = np.triu_indices(N, k=1)
    weights = sym_matrix[triu_indices]

    if k > len(weights): k = len(weights)
    top_indices = np.argsort(weights)[-k:][::-1]

    edge_set = set()
    rows = triu_indices[0]
    cols = triu_indices[1]

    for idx in top_indices:
        u, v = rows[idx], cols[idx]
        edge_set.add(tuple(sorted((u, v))))

    return edge_set


def plot_differential_connectome(unique_pat, unique_hc, coords, title, filename):
    """
    Core plotting function:
    - unique_pat (Set of tuples): Edges unique to Patient -> Red (value set to 1)
    - unique_hc (Set of tuples): Edges unique to HC -> Blue (value set to -1)
    """
    N = coords.shape[0]
    adj_plot = np.zeros((N, N))

    # Fill matrix
    # HC = Blue (-1), Patient = Red (+1)
    for u, v in unique_hc:
        adj_plot[u, v] = -1
        adj_plot[v, u] = -1

    for u, v in unique_pat:
        adj_plot[u, v] = 1
        adj_plot[v, u] = 1

    fig = plt.figure(figsize=(10, 8))

    # Plot using nilearn
    # edge_cmap='coolwarm': Negative values blue, positive values red, 0 is white (transparent)
    display = plotting.plot_connectome(
        adjacency_matrix=adj_plot,
        node_coords=coords,
        node_size=15,  # Node size
        node_color='black',  # Node color
        edge_cmap='coolwarm',
        edge_vmin=-1,
        edge_vmax=1,
        edge_threshold=0.5,  # Threshold set to 0.5 to ensure only edges with 1 and -1 are shown, filtering out 0
        colorbar=False,  # Can be turned off, manual legend is clearer
        display_mode='lzry',  # Display left, top, right, bottom views
        figure=None
    )

    fig = display.frame_axes.figure

    # Add title and background box
    fig.suptitle(f"{title}", fontsize=12, x=0.01, y=0.99,
                 verticalalignment='top', horizontalalignment='left', color='white',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='black', edgecolor='none'))

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Patient Unique'),
        Line2D([0], [0], color='blue', lw=2, label='HC Unique')
    ]

    # Place legend in the upper right corner
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.99),
               facecolor='black', labelcolor='white', edgecolor='none')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Saved: {os.path.basename(filename)}")


def main():
    set_seed(SEED)
    device = ARGS['device']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== Group Differential Plotting | Task: {ARGS['task']} ===")

    # 1. Load Data
    print("Loading Data...")
    raw_data = load_data(DATA_PATH)
    processed_dict = preprocess_labels(raw_data, task=ARGS['task'])

    # 2. 1:1 Balance
    group_hc, group_patient = get_balanced_data(processed_dict)

    # 3. Prepare Coordinates
    # Assume all samples have the same number of nodes
    N = group_hc[0]['SC'].shape[0]
    coords = load_glasser_coords(COORD_FILE, N)

    # 4. Load Model
    print("Loading Model...")
    model = DIALModel(
        N=N, d_model=ARGS['d_model'], num_classes=2, task='classification',
        num_node_layers=ARGS['num_node_layers'], num_graph_layers=ARGS['num_graph_layers'],
        dropout=ARGS['dropout']
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: Model weights not found! Using random weights.")

    # 5. Compute Average Matrices
    print("Computing Group Averages...")
    hc_sc, hc_fc, hc_model = get_averaged_matrices(group_hc, model, device, N)
    pat_sc, pat_fc, pat_model = get_averaged_matrices(group_patient, model, device, N)

    # 6. Plotting Loop
    print(f"Plotting Differential Connectomes (Top-{TOP_K})...")

    tasks = [
        ('SC', hc_sc, pat_sc),
        ('FC', hc_fc, pat_fc),
        ('Model_Information_Load', hc_model, pat_model)
    ]

    for name, mat_hc, mat_pat in tasks:
        if mat_hc is None: continue

        # Extract Top-K sets
        set_hc = get_topk_edge_set(mat_hc, TOP_K)
        set_pat = get_topk_edge_set(mat_pat, TOP_K)

        # Calculate unique edges
        unique_hc = set_hc - set_pat
        unique_pat = set_pat - set_hc

        print(f"[{name}] HC Unique: {len(unique_hc)} | Patient Unique: {len(unique_pat)}")

        # Plot
        plot_differential_connectome(
            unique_pat, unique_hc, coords,
            title=f"{ARGS['task']} Group Difference: {name}\nTop-{TOP_K} Edges",
            filename=os.path.join(OUTPUT_DIR, f"{ARGS['task']}_Diff_{name}.png")
        )

    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()