import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from dial.model import DIALModel
from dial.data import (
    ABCDDataset,
    load_data,
    preprocess_labels,
    balance_dataset,
    split_dataset
)


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: str = 'cpu',
                epoch: int = 0,
                num_epochs: int = 1,
                show_progress: bool = True) -> Dict:
    model.train()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    sample_count = 0

    batch_iterable = dataloader
    if show_progress:
        batch_iterable = tqdm(
            dataloader,
            desc=f"Train {epoch + 1}/{num_epochs}",
            leave=False
        )

    for batch in batch_iterable:
        labels = batch['labels'].to(device).squeeze(-1).long()
        batch_size = labels.shape[0]
        node_feat = batch['node_feat'].to(device)
        in_degree = batch['in_degree'].to(device)
        out_degree = batch['out_degree'].to(device)
        path_data = batch['path_data'].to(device)
        dist = batch['dist'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        S = batch['S'].to(device)
        F = batch['F'].to(device)

        optimizer.zero_grad()
        y_pred, loss = model(
            node_feat=node_feat,
            in_degree=in_degree,
            out_degree=out_degree,
            path_data=path_data,
            dist=dist,
            attn_mask=attn_mask,
            S=S,
            F=F,
            y=labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        preds = y_pred.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())
        sample_count += batch_size

    if show_progress and hasattr(batch_iterable, 'close'):
        batch_iterable.close()

    avg_loss = total_loss / max(sample_count, 1)
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: str = 'cpu') -> Dict:
    """
    Evaluate the model on the provided dataloader and compute metrics.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader that yields mini-batches.
        device: Device identifier.

    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC, raw predictions, and labels.
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device).squeeze(-1).long()
            names = batch['names']
            y_pred, _ = model.inference(
                node_feat=batch['node_feat'].to(device),
                in_degree=batch['in_degree'].to(device),
                out_degree=batch['out_degree'].to(device),
                path_data=batch['path_data'].to(device),
                dist=batch['dist'].to(device),
                attn_mask=batch['attn_mask'].to(device),
                S=batch['S'].to(device),
                F=batch['F'].to(device),
                k=100
            )
            probs = torch.softmax(y_pred, dim=1)

            all_preds.extend(y_pred.argmax(dim=1).detach().cpu().tolist())
            all_probs.extend(probs[:, 1].detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_names.extend(names)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'names': all_names
    }

    return metrics


def print_metrics(metrics: Dict, prefix: str = ""):
    """Print formatted metrics with an optional text prefix."""
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1-Score:  {metrics['f1']:.4f}")
    print(f"{prefix}AUC:       {metrics['auc']:.4f}")
    if 'confusion_matrix' in metrics:
        print(f"{prefix}Confusion Matrix:")
        print(metrics['confusion_matrix'])


def main(
        data_path: str,
        task: str = 'OCD',
        output_dir: str = './results',
        # Model hyperparameters
        d_model: int = 64,
        num_node_layers: int = 2,
        num_graph_layers: int = 2,
        # Training hyperparameters
        num_epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        # Dataset options
        test_size: float = 0.3,
        balance_ratio: float = 1.0,
        random_state: int = 42,
        # Device
        device: str = 'cpu'
):
    print("=" * 100)
    print(f"DIAL experiment - task: {task}")
    print("=" * 100)

    os.makedirs(output_dir, exist_ok=True)
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    data_dict = load_data(data_path)

    print(f"\nLabel preprocessing - {task}")
    processed_dict = preprocess_labels(data_dict, task=task)

    print(f"\nBalance dataset ({balance_ratio}:1 ratio)")
    balanced_dict = balance_dataset(processed_dict, ratio=balance_ratio, random_state=random_state)

    print(f"\nSplit dataset (test size {test_size})")
    train_data, test_data = split_dataset(balanced_dict, test_size=test_size, random_state=random_state)
    # train_data = train_data[:16]
    # test_data = test_data[:8]

    train_dataset = ABCDDataset(train_data, device=device)
    test_dataset = ABCDDataset(test_data, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate
    )

    N = train_data[0]['SC'].shape[0]
    print(f"Number of nodes: {N}")

    print("\n[Step 6] Build DIAL model")
    model = DIALModel(
        N=N,
        d_model=d_model,
        num_classes=2,
        task='classification',
        num_node_layers=num_node_layers,
        num_graph_layers=num_graph_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print("-" * 100)
    print(f"\nTrain for {num_epochs} epochs")
    print("-" * 100)

    best_f1 = 0.0
    best_epoch = 0
    train_history = []
    test_history = []

    for epoch in range(num_epochs):
        # Training
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs
        )

        # Evaluation
        test_metrics = evaluate(model, test_loader, device)

        # Tracking
        train_history.append(train_metrics)
        test_history.append(test_metrics)

        # Scheduler step
        scheduler.step(test_metrics['f1'])

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(
                f"  Test  - Acc: {test_metrics['accuracy']:.4f}, "
                f"P: {test_metrics['precision']:.4f}, "
                f"R: {test_metrics['recall']:.4f}, "
                f"F1: {test_metrics['f1']:.4f}, "
                f"AUC: {test_metrics['auc']:.4f}"
            )

        # Save best checkpoint
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_epoch = epoch
            model_path = os.path.join(task_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  *** Saved new best model (F1={best_f1:.4f}) ***")

    # 9. Final evaluation with best checkpoint
    print(f"\n[Step 8] Final evaluation (best epoch: {best_epoch + 1})")
    print("-" * 80)

    model.load_state_dict(torch.load(os.path.join(task_dir, 'best_model.pth')))

    print("\nTrain results:")
    train_final = evaluate(model, train_loader, device)
    print_metrics(train_final, prefix="  ")

    print("\nTest results:")
    test_final = evaluate(model, test_loader, device)
    print_metrics(test_final, prefix="  ")

    # 10. Save artifacts
    print(f"\n[Step 9] Save artifacts to {task_dir}")

    results = {
        'task': task,
        'best_epoch': best_epoch,
        'train_final': train_final,
        'test_final': test_final,
        'train_history': train_history,
        'test_history': test_history,
        'config': {
            'N': N,
            'd_model': d_model,
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'test_size': test_size,
            'balance_ratio': balance_ratio,
        }
    }

    with open(os.path.join(task_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Save classification report
    with open(os.path.join(task_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"DIAL experiment summary - {task}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best epoch: {best_epoch + 1}\n\n")
        f.write("Test results:\n")
        f.write(classification_report(
            test_final['labels'],
            test_final['predictions'],
            target_names=['Negative', 'Positive']
        ))

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DIAL brain disorder classification experiment')

    # Data
    parser.add_argument('--data_path', type=str, default=r"W:\Brain Analysis\data\ABCD\processed\data_dict.pkl")
    parser.add_argument('--task', type=str, default='OCD', choices=['OCD', 'ADHD_ODD_Cond'], help='Task name')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.3, help='Hold-out test fraction')
    parser.add_argument('--balance_ratio', type=float, default=1.0, help='Negative-to-positive balance ratio')
    parser.add_argument('--random_state', type=int, default=20010920, help='Random seed')

    # Model
    parser.add_argument('--d_model', type=int, default=64, help='Transformer hidden dimension')
    parser.add_argument('--num_node_layers', type=int, default=2, help='Number of node encoder layers')
    parser.add_argument('--num_graph_layers', type=int, default=2, help='Number of graph Transformer layers')

    # Training
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay factor')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu/cuda)')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    main(
        data_path=args.data_path,
        task=args.task,
        output_dir=args.output_dir,
        d_model=args.d_model,
        num_node_layers=args.num_node_layers,
        num_graph_layers=args.num_graph_layers,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        test_size=args.test_size,
        balance_ratio=args.balance_ratio,
        random_state=args.random_state,
        device=args.device
    )
