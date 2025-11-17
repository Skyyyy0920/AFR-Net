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
                device: str = 'cpu') -> Dict:
    model.train()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    sample_count = 0

    for batch in dataloader:
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
    评估模型
    
    参数:
        model: 模型
        dataset: 数据集
        device: 设备
        
    返回:
        metrics: 评估指标字典
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

    # 计算指标
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
    """打印指标"""
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
        # 模型参数
        d_model: int = 64,
        num_node_layers: int = 2,
        num_graph_layers: int = 2,
        # 训练参数
        num_epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 4,
        # 数据参数
        test_size: float = 0.3,
        balance_ratio: float = 1.0,
        random_state: int = 42,
        # 设备
        device: str = 'cpu'
):
    """
    主实验函数
    
    参数:
        data_path: 数据文件路径
        task: 任务类型 'OCD' 或 'ADHD_ODD_Cond'
        output_dir: 输出目录
        d_model: 模型维度
        num_node_layers: 节点编码器层数
        num_graph_layers: 图Transformer层数
        num_epochs: 训练轮数
        lr: 学习率
        batch_size: DataLoader批大小
        test_size: 测试集比例
        balance_ratio: 正负样本比例
        random_state: 随机种子
        device: 设备
    """
    print("=" * 80)
    print(f"DIAL模型实验 - 任务: {task}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    # 1. 加载数据
    print("\n[步骤1] 数据加载")
    data_dict = load_data(data_path)

    # 2. 预处理标签
    print(f"\n[步骤2] 标签预处理 - {task}")
    processed_dict = preprocess_labels(data_dict, task=task)

    # 3. 平衡数据集
    print(f"\n[步骤3] 数据平衡 (比例 {balance_ratio}:1)")
    balanced_dict = balance_dataset(processed_dict, ratio=balance_ratio, random_state=random_state)

    # 4. 划分数据集
    print(f"\n[步骤4] 数据划分 (测试集比例 {test_size})")
    train_data, test_data = split_dataset(balanced_dict, test_size=test_size, random_state=random_state)
    train_data = train_data[:16]
    test_data = test_data[:8]

    # 5. 创建数据集
    print("\n[步骤5] 创建数据集对象")
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

    # 获取节点数（假设所有样本的节点数相同）
    N = train_data[0]['SC'].shape[0]
    print(f"节点数: {N}")

    # 6. 创建模型
    print("\n[步骤6] 创建DIAL模型")
    model = DIALModel(
        N=N,
        d_model=d_model,
        num_classes=2,
        task='classification',
        num_node_layers=num_node_layers,
        num_graph_layers=num_graph_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 7. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # 8. 训练
    print(f"\n[步骤7] 开始训练 ({num_epochs} epochs)")
    print("-" * 80)

    best_f1 = 0.0
    best_epoch = 0
    train_history = []
    test_history = []

    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # 评估
        test_metrics = evaluate(model, test_loader, device)

        # 记录
        train_history.append(train_metrics)
        test_history.append(test_metrics)

        # 学习率调度
        scheduler.step(test_metrics['f1'])

        # 打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  测试 - Acc: {test_metrics['accuracy']:.4f}, P: {test_metrics['precision']:.4f}, "
                  f"R: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")

        # 保存最佳模型
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_epoch = epoch
            model_path = os.path.join(task_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  *** 保存最佳模型 (F1={best_f1:.4f}) ***")

    # 9. 加载最佳模型并最终评估
    print(f"\n[步骤8] 最终评估 (最佳epoch: {best_epoch + 1})")
    print("-" * 80)

    model.load_state_dict(torch.load(os.path.join(task_dir, 'best_model.pth')))

    print("\n训练集表现:")
    train_final = evaluate(model, train_loader, device)
    print_metrics(train_final, prefix="  ")

    print("\n测试集表现:")
    test_final = evaluate(model, test_loader, device)
    print_metrics(test_final, prefix="  ")

    # 10. 保存结果
    print(f"\n[步骤9] 保存结果到 {task_dir}")

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
            'test_size': test_size,
            'balance_ratio': balance_ratio,
        }
    }

    with open(os.path.join(task_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # 保存分类报告
    with open(os.path.join(task_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"DIAL模型实验结果 - {task}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"最佳Epoch: {best_epoch + 1}\n\n")
        f.write("测试集表现:\n")
        f.write(classification_report(
            test_final['labels'],
            test_final['predictions'],
            target_names=['Negative', 'Positive']
        ))

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DIAL模型脑疾病分类实验')

    # data
    parser.add_argument('--data_path', type=str, default=r"W:\Brain Analysis\data\ABCD\processed\data_dict.pkl")
    parser.add_argument('--task', type=str, default='OCD', choices=['OCD', 'ADHD_ODD_Cond'], help='任务类型')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.3, help='测试集比例')
    parser.add_argument('--balance_ratio', type=float, default=1.0, help='正负样本比例')
    parser.add_argument('--random_state', type=int, default=20010920, help='随机种子')

    # model
    parser.add_argument('--d_model', type=int, default=64, help='模型维度')
    parser.add_argument('--num_node_layers', type=int, default=2, help='节点编码器层数')
    parser.add_argument('--num_graph_layers', type=int, default=2, help='图Transformer层数')

    # training
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')

    # device
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cpu/cuda)')

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
        batch_size=args.batch_size,
        test_size=args.test_size,
        balance_ratio=args.balance_ratio,
        random_state=args.random_state,
        device=args.device
    )
