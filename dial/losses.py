"""
损失函数模块
实现任务损失和各种正则化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def compute_losses(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    L: torch.Tensor,
    m: torch.Tensor,
    task: str,
    S_e: torch.Tensor,
    a_e: torch.Tensor = None,
    lambda_align: float = 0.2,
    lambda_budget: float = 0.05,
    lambda_gate: float = 1e-4,
    eps: float = 1e-9
) -> Dict[str, torch.Tensor]:
    """
    计算总损失和各项分损失
    
    损失组成:
    1. 任务损失 (分类/回归)
    2. 对齐损失 (载荷L与掩码m的一致性)
    3. 预算损失 (结构成本约束)
    4. 门稀疏损失 (门值正则化)
    
    参数:
        y_pred: 预测值
            - 分类: [num_classes] logits
            - 回归: 标量
        y: 真实标签
            - 分类: 标量整数
            - 回归: 标量浮点数
        L: [E] 边载荷
        m: [E] 软掩码
        task: 任务类型 'classification' 或 'regression'
        S_e: [E] 边的结构连接强度
        a_e: [E] 边门值（可选）
        lambda_align: 对齐损失系数
        lambda_budget: 预算损失系数
        lambda_gate: 门稀疏损失系数
        eps: 数值稳定性参数
        
    返回:
        loss_dict: 包含各项损失的字典
            - 'loss': 总损失
            - 'task': 任务损失
            - 'align': 对齐损失
            - 'budget': 预算损失
            - 'gate': 门稀疏损失
    """
    
    # 1. 任务损失
    if task == 'classification':
        L_task = F.cross_entropy(y_pred.unsqueeze(0), y.unsqueeze(0))
    elif task == 'regression':
        L_task = F.mse_loss(y_pred, y)
    else:
        raise ValueError(f"不支持的任务类型: {task}")
    
    # 2. 对齐损失（载荷L与掩码m的一致性）
    # p = softmax(L), q = softmax(log(m + eps))
    # L_align = KL(p || stopgrad(q))
    p = F.softmax(L, dim=0)  # [E]
    log_m = torch.log(m + eps)
    q = F.softmax(log_m, dim=0)  # [E]
    
    # KL散度: KL(p || q) = sum(p * log(p/q))
    with torch.no_grad():
        q_detach = q.detach()
    
    L_align = F.kl_div(
        torch.log(q_detach + eps),
        p,
        reduction='batchmean',
        log_target=False
    )
    
    # 3. 预算损失（结构成本约束）
    # c_e = -log(S_e + eps)
    c_e = -torch.log(S_e + eps)  # [E]
    L_budget = (c_e * m).mean()
    
    # 4. 门稀疏损失
    if a_e is not None:
        L_gate = a_e.abs().mean()
    else:
        L_gate = torch.tensor(0.0, device=L.device)
    
    # 总损失
    loss = L_task + lambda_align * L_align + lambda_budget * L_budget + lambda_gate * L_gate
    
    # 返回损失字典
    loss_dict = {
        'loss': loss,
        'task': L_task.item(),
        'align': L_align.item(),
        'budget': L_budget.item(),
        'gate': L_gate.item() if isinstance(L_gate, torch.Tensor) else L_gate
    }
    
    return loss_dict


def classification_loss(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    分类任务损失
    
    参数:
        y_pred: [num_classes] logits
        y: 标量整数标签
        
    返回:
        loss: 标量
    """
    return F.cross_entropy(y_pred.unsqueeze(0), y.unsqueeze(0))


def regression_loss(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    回归任务损失
    
    参数:
        y_pred: 标量预测值
        y: 标量真实值
        
    返回:
        loss: 标量
    """
    return F.mse_loss(y_pred.unsqueeze(0), y.unsqueeze(0))


def alignment_loss(
    L: torch.Tensor,
    m: torch.Tensor,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    对齐损失：确保载荷L和掩码m的一致性
    
    使用KL散度测量两个分布的差异
    
    参数:
        L: [E] 边载荷
        m: [E] 软掩码
        eps: 数值稳定性参数
        
    返回:
        loss: 标量
    """
    # 将L和m转换为概率分布
    p = F.softmax(L, dim=0)
    log_m = torch.log(m + eps)
    q = F.softmax(log_m, dim=0)
    
    # KL(p || stopgrad(q))
    with torch.no_grad():
        q_detach = q.detach()
    
    loss = F.kl_div(
        torch.log(q_detach + eps),
        p,
        reduction='batchmean',
        log_target=False
    )
    
    return loss


def budget_loss(
    m: torch.Tensor,
    S_e: torch.Tensor,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    预算损失：惩罚选择结构成本高的边
    
    参数:
        m: [E] 软掩码
        S_e: [E] 边的结构连接强度
        eps: 数值稳定性参数
        
    返回:
        loss: 标量
    """
    # 结构成本 c_e = -log(S_e + eps)
    c_e = -torch.log(S_e + eps)
    
    # 期望成本
    loss = (c_e * m).mean()
    
    return loss


def gate_sparsity_loss(a_e: torch.Tensor) -> torch.Tensor:
    """
    门稀疏损失：鼓励门值稀疏
    
    参数:
        a_e: [E] 边门值
        
    返回:
        loss: 标量
    """
    return a_e.abs().mean()

