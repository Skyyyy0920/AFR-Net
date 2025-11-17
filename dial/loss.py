"""
损失函数模块
实现任务损失和各种正则化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


def compute_losses(
        y_pred: torch.Tensor,  # [B, num_classes] or [B]
        y: torch.Tensor,  # [B]
        L_list: List[torch.Tensor],  # List of [E_i]
        m_list: List[torch.Tensor],  # List of [E_i]
        task: str,
        S_e_list: List[torch.Tensor],  # List of [E_i]
        a_e_list: List[torch.Tensor],
        lambda_align: float = 1,
        lambda_budget: float = 1,
        lambda_gate: float = 0.1,
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
    B = len(L_list)

    if task == 'classification':
        L_task = F.cross_entropy(y_pred, y)
    elif task == 'regression':
        L_task = F.mse_loss(y_pred, y)
    else:
        raise ValueError(f"不支持的任务类型: {task}")

    L_align_sum = 0.0
    L_budget_sum = 0.0
    L_gate_sum = 0.0

    for i in range(B):
        L = L_list[i]
        m = m_list[i]
        S_e = S_e_list[i]
        a_e = a_e_list[i]

        # 对齐损失（载荷L与掩码m的一致性）
        # p = softmax(L), q = softmax(log(m + eps))
        # L_align = KL(p || stopgrad(q))
        p = F.softmax(L, dim=0)
        q = F.softmax(torch.log(m + eps), dim=0)

        q_detach = q.detach()

        L_align = F.kl_div(
            torch.log(q_detach + eps),
            p,
            reduction='batchmean',
            log_target=False
        )

        # 预算损失（结构成本约束）
        c_e = -torch.log(S_e + eps)
        L_budget = (c_e * m).mean()

        # 门稀疏损失
        L_gate = a_e.abs().mean()

        L_align_sum += L_align
        L_budget_sum += L_budget
        L_gate_sum += L_gate

    L_align_avg = L_align_sum / B
    L_budget_avg = L_budget_sum / B
    L_gate_avg = L_gate_sum / B

    loss = L_task + lambda_align * L_align_avg + lambda_budget * L_budget_avg + lambda_gate * L_gate_avg

    loss_dict = {
        'loss': loss,
        'task': L_task.item(),
        'align': L_align_avg.item(),
        'budget': L_budget_avg.item(),
        'gate': L_gate_avg.item()
    }

    return loss_dict
