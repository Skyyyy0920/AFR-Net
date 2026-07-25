# AFR-Net: Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing

Official PyTorch implementation of our **ICML 2026** paper
[**Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing**](https://github.com/Skyyyy0920/AFR-Net).

*Tianhao Huang, Guanghui Min, Zhenyu Lei, Aiying Zhang, Chen Chen — University of Virginia.*
*Proceedings of the 43rd International Conference on Machine Learning (ICML), Seoul, PMLR 306, 2026.*

---

## Overview

Deciphering how macroscopic cognitive phenotypes emerge from microscopic neuronal connectivity requires
jointly modeling **structural connectivity (SC)** — the hard-wired anatomical infrastructure — and
**functional connectivity (FC)** — the statistical traffic of neural activity. Existing multi-modal fusion
methods operate mostly at a **topological or architectural** level: they model *which* structural pathways
support a functional link, but cannot quantify *how much* information is actually routed along each pathway.
This leaves them unable to fully explain the regional **coupling–heterogeneity** of SC and FC.

**AFR-Net (Adaptive Flow Routing Network)** reformulates SC–FC fusion through the lens of **neural
communication dynamics**. It is a physics-informed framework that treats the structural connectome as a
**dynamic flow network** — anatomical tracts are wires with learnable conductances, and every functionally
coupled pair of regions injects a unit of "current". Solving this circuit yields, for each edge, the total
information traffic it must carry to satisfy all functional demands. These latent communication patterns are
then used to guide message passing for downstream diagnosis, delivering both **state-of-the-art accuracy**
and **interpretable discovery of critical neural pathways**.

<p align="center">
  <em>Neural communication is modeled as an intermediate regime between stochastic diffusion and
  deterministic shortest-path routing, using multiple alternative pathways for robustness and efficiency.</em>
</p>

### Method at a glance

AFR-Net fuses SC and FC through three integrated stages:

1. **Physics-Informed Graph Construction.** Node representations are initialized with a structure-aware flow
   encoder that replaces shortest-path heuristics with **effective resistance distance (ERD)** computed on the
   weighted structural Laplacian, injected as an attention bias. A learnable **edge-gating network** predicts a
   strictly positive flow capacity `c_ij = exp(ω_gate(h_i, h_j))` for every structural edge.

2. **Differentiable Information Flow Solver.** Modeling the brain as a resistor network, AFR-Net derives a
   **closed-form** solution for edge-level information traffic driven by global functional demands:

   ```
   Φ_ij = 2 · c_ij · (e_i − e_j)ᵀ · L_flow⁻¹ · L_fc · L_flow⁻¹ · (e_i − e_j)
   ```

   where `L_flow = Bᵀ C B + δI` is the regularized structural Laplacian and `L_fc = D_fc − |A_fc|` is the
   functional-demand Laplacian. High `Φ_ij` marks critical transmission pathways — both direct highways and
   essential structural detours. Backpropagation uses **implicit differentiation** (the adjoint method), keeping
   the same complexity class as the forward solve.

3. **Pattern-Guided Aggregation.** The learned flow map `Φ` is converted into a soft routing mask via a
   differentiable Log-Min-Max normalization and used to bias a masked Transformer's self-attention, so message
   passing is guided by the discovered communication patterns before pooling and classification.

---

## Key Results

AFR-Net is evaluated on two multi-modal neuroimaging benchmarks — **ABCD** (OCD, ADHD, Anxiety binary
classification) and **PPMI** (Parkinson's Disease vs. Healthy Control) — against 14 baselines (6 general
graph-learning backbones and 8 specialized brain-network models), all receiving both SC and FC.

| Method       | OCD F1 | OCD AUC | ADHD F1 | ADHD AUC | Anx F1 | Anx AUC | PPMI F1 | PPMI AUC |
| :----------- | :----: | :-----: | :-----: | :------: | :----: | :-----: | :-----: | :------: |
| RH-BrainFS   | 59.55  | 59.69   | 61.59   | 61.76    | 53.69  | 54.75   | 68.54   | 68.25    |
| NeuroPath    | 64.13  | 69.39   | 62.36   | **64.19**| 55.56  | 57.65   | 74.56   | 71.88    |
| **AFR-Net**  | **70.62** | **72.68** | **65.50** | 62.38 | **62.15** | **59.03** | **83.70** | **90.38** |

AFR-Net ranks first on the large majority of metrics across all four tasks (mean over 3 seeds; full five-metric
tables are in the paper appendix). Beyond accuracy, its interpretability analysis autonomously recovers the
visual/somatomotor **structural core** of brain communication and surfaces disease-specific pathological hubs
(e.g. hyper-active hippocampus–DMN routing in anxiety, visual–limbic hyper-connectivity in OCD) that align with
established neuroscientific findings.

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
