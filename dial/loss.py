"""
Loss functions for the DIAL model, covering both task loss and regularizers.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List


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
    Compute the total loss and its decomposition.

    Loss terms:
        1. Task loss (classification or regression).
        2. Alignment loss (consistency between load L and mask m).
        3. Budget loss (structural cost constraint).
        4. Gate sparsity loss (penalizing gate magnitudes).

    Args:
        y_pred: Predictions (logits for classification, scalar for regression).
        y: Ground-truth labels.
        L_list: Edge load tensors per sample.
        m_list: Soft mask tensors per sample.
        task: Either 'classification' or 'regression'.
        S_e_list: Structural edge strengths per sample.
        a_e_list: Edge gate outputs per sample.
        lambda_align: Alignment loss weight.
        lambda_budget: Budget loss weight.
        lambda_gate: Gate sparsity loss weight.
        eps: Numerical stability constant.

    Returns:
        Dictionary with the total loss as well as each component.
    """
    B = len(L_list)

    if task == 'classification':
        L_task = F.cross_entropy(y_pred, y)
    elif task == 'regression':
        L_task = F.mse_loss(y_pred, y)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    L_align_sum = 0.0
    L_budget_sum = 0.0
    L_gate_sum = 0.0

    for i in range(B):
        L = L_list[i]
        m = m_list[i]
        S_e = S_e_list[i]
        a_e = a_e_list[i]

        # Alignment loss between load L and mask m.
        # p = softmax(L), q = softmax(log(m + eps))
        # L_align = KL(p || stopgrad(q))
        p = F.softmax(L, dim=0)
        q = F.softmax(torch.log(m + eps), dim=0)
        q_detach = q.detach()
        L_align = F.kl_div(  # TODO: 需要吗？
            torch.log(q_detach + eps),
            p,
            reduction='batchmean',
            log_target=False
        )

        # Budget loss (structural cost constraint).
        c_e = -torch.log(S_e + eps)
        L_budget = (c_e * m).mean()

        # Gate sparsity loss.
        L_gate = a_e.abs().mean()

        # L_sparse = torch.mean(torch.stack([a_e.mean() for a_e in a_e_list]))  # TODO

        L_align_sum += L_align
        L_budget_sum += L_budget
        L_gate_sum += L_gate

    L_align_avg = L_align_sum / B
    L_budget_avg = L_budget_sum / B
    L_gate_avg = L_gate_sum / B

    loss = L_task + lambda_align * L_align_avg + lambda_budget * L_budget_avg + lambda_gate * L_gate_avg

    return {
        'loss': loss,
        'task': L_task.item(),
        'align': L_align_avg.item(),
        'budget': L_budget_avg.item(),
        'gate': L_gate_avg.item()
    }
