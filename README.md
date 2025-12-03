# DIAL: Detour-aware Information-load Aggregation in Brain Networks

This is the code implementation for our paper [**]().

## Cite our work
```

```


## Project Structure

```text
DIAL/
├── baselines/           # Baseline implementations (MLP, GCN, GAT, Graphormer, etc.)
│   ├── models.py        # Baseline model architectures
│   ├── run.py           # Multi-seed entry point for baselines
│   └── run_baselines.py # Single-run logic for baselines
├── data/                # Data storage and preprocessing scripts
│   └── PPMI/            # PPMI-specific processing scripts
├── dial/                # DIAL core source code
│   ├── data.py          # Data loading & Graph construction (ABCDDataset, PPMIDataset)
│   ├── loss.py          # Loss functions
│   ├── model.py         # DIAL Architecture (Node Encoder, EdgeGate, MaskedTransformer)
│   ├── routing.py       # Energy computation & Routing logic
│   └── utils.py         # Utilities for Graph Laplacian, etc.
├── main.py              # Single-run entry point for DIAL
├── run.py               # Multi-seed experiment & metrics aggregation for DIAL
└── requirements.txt     # Python dependencies
```


## Requirements
```
pip install -r requirements.txt
```


## Data Preparation

Data should be saved as a `pickle` (`.pkl`) file containing a dictionary. The keys are sample IDs, and the values are dictionaries with the following fields:

  - `SC`: Structural Connectivity Matrix (N x N)
  - `FC`: Functional Connectivity Matrix (N x N)
  - `label`: Classification label (0 or 1)


Before running, please modify the `main.py` defaults or pass the path via `--data_path` arguments.


## Usage


### Training DIAL

Use the `run.py` script to automatically run experiments across multiple random seeds and report the aggregated metrics (Mean ± Std).

```bash
python run.py \
  --task OCD \
  --data_path /path/to/your/data_dict.pkl \
  --device cuda:0 \
  --seeds 0 1 2 3 4
```

### Training Baselines

We provide dual-stream implementations for MLP, GCN, GAT, and Graphormer.

```bash
python -m baselines.run \
  --task OCD \
  --models mlp,gcn,gat,graphormer \
  --seeds 0 1 2 3 4
```

### Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--task` | Downstream task name (e.g., `OCD`, `Dep`, `PPMI`) | `OCD` |
| `--d_model` | Hidden dimension for Transformer/Graphormer | `64` |
| `--num_epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size | `64` |
| `--lr` | Learning rate | `5e-4` |
| `--test_size` | Test set fraction (for non-PPMI data) | `0.3` |


## Results

After running the scripts, results will be saved in the `./results/<task>/<timestamp>/` directory.

  - `experiment.log`: Detailed training logs.
  - `results.pkl`: Dictionary containing training/validation/testing history and final metrics.
  - `classification_report.txt`: Detailed classification report for the best model.
  - `best_model.pth`: Model weights with the best validation AUC.


## FAQ
- Q:  
  A: 
