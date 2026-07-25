# AFR-Net: Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing

Official PyTorch implementation of our **ICML 2026** paper
[**Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing**](https://openreview.net/forum?id=clFn9vQ8cK).

---

## Project Structure

```text
AFR-Net/
├── dial/                    # AFR-Net core implementation
│   ├── data.py              # Data loading & graph construction (ABCDDataset, PPMIDataset)
│   ├── model.py             # Architecture: node encoder, edge gate, masked Transformer
│   ├── routing.py           # Differentiable information flow solver & routing logic
│   ├── loss.py              # Loss functions
│   └── utils.py             # Graph Laplacian / effective-resistance utilities
├── baselines/               # Dual-stream baseline implementations
│   ├── models.py            # Baseline architectures (MLP, GCN, GAT, Graphormer, ...)
│   ├── run.py               # Multi-seed entry point for baselines
│   └── run_baselines.py     # Single-run logic for baselines
├── data/                    # Data storage and preprocessing
│   └── PPMI/                # PPMI-specific processing scripts
├── plot/                    # Interpretability & analysis figures
├── main.py                  # Single-run entry point for AFR-Net
├── run.py                   # Multi-seed experiment & metrics aggregation for AFR-Net
├── circle_plot.py           # Group-difference connectome circle plots
├── test.py                  # Evaluation helpers
└── requirements.txt         # Python dependencies
```

> The core package directory is named `dial/` (the project's internal codename); it contains the full AFR-Net model.

---

## Requirements

```bash
pip install -r requirements.txt
```

The framework is implemented with **PyTorch** and the **Deep Graph Library (DGL)**. Experiments in the paper
were run on an NVIDIA A100 GPU.

---

## Data Preparation

Each dataset is a `pickle` (`.pkl`) file containing a dictionary. Keys are sample IDs; each value is a
dictionary with the following fields:

- `SC`: Structural Connectivity matrix (`N × N`)
- `FC`: Functional Connectivity matrix (`N × N`)
- `label`: Classification label (`0` or `1`)

For ABCD, the connectome is parcellated with the HCP-MMP1.0 (Glasser) / FreeSurfer atlas; for PPMI, with the
AAL atlas (90 ROIs). Before running, either edit the defaults in `main.py` / `run.py` or pass the path via the
`--data_path` argument (PPMI uses `--ppmi_train_path` / `--ppmi_test_path`).

---

## Usage

### Training AFR-Net

Use `run.py` to run experiments across multiple random seeds and report aggregated metrics (mean ± std):

```bash
python run.py \
  --task OCD \
  --data_path /path/to/your/data_dict.pkl \
  --device cuda:0 \
  --seeds 0 1 2
```

For a single run, use `main.py` with the same arguments and `--random_seed`.

### Training Baselines

Dual-stream adaptations (each backbone encodes SC and FC separately, then concatenates before the head):

```bash
python -m baselines.run \
  --task OCD \
  --models mlp,gcn,gat,graphormer \
  --seeds 0 1 2
```

### Key Arguments

| Argument            | Description                                             | Default  |
| :------------------ | :----------------------------------------------------- | :------- |
| `--task`            | Task name (`OCD`, `ADHD`, `Anx`, `PPMI`, ...)          | `OCD`    |
| `--data_path`       | Path to the `.pkl` data dictionary                     | —        |
| `--d_model`         | Hidden dimension                                       | `64`     |
| `--num_node_layers` | Number of node-encoder (Transformer) layers            | `2`      |
| `--num_graph_layers`| Number of masked-Transformer graph layers              | `2`      |
| `--num_epochs`      | Number of training epochs                              | `50`     |
| `--batch_size`      | Batch size                                             | `64`     |
| `--lr`              | Learning rate                                          | `5e-4`   |
| `--weight_decay`    | Weight decay                                           | `1e-3`   |
| `--dropout`         | Dropout rate                                           | `0.3`    |
| `--test_size`       | Hold-out test fraction (non-PPMI)                      | `0.3`    |
| `--device`          | Device (`cpu` / `cuda`)                                | `cuda`   |

Data are split into train/val/test with a 6:1:3 ratio. Additional architectural hyper-parameters used in the
paper: ERD encoder = 2-layer MLP (hidden 128, GELU); edge-gating net = 2-layer MLP (hidden 64, SiLU);
Laplacian regularizer `δ = 1e-6`; mask scaling `τ = 8.0`.

---

## Results & Outputs

After training, results are saved under `./results/<task>/<timestamp>/`:

- `experiment.log` — detailed training logs
- `results.pkl` — dictionary with train/val/test history and final metrics
- `classification_report.txt` — classification report for the best model
- `best_model.pth` — model weights with the best validation AUC

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2026afrnet,
  title     = {Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing},
  author    = {Huang, Tianhao and Min, Guanghui and Lei, Zhenyu and Zhang, Aiying and Chen, Chen},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR}
}
```

## Acknowledgements

This work was supported in part by the Commonwealth Cyber Initiative (CCI) under Award No. VV-1Q26-005 and by
the National Science Foundation under Grant No. 2331315.
