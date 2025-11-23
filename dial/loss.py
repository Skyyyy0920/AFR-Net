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
    if task == 'classification':
        L_task = F.cross_entropy(y_pred, y)
    elif task == 'regression':
        L_task = F.mse_loss(y_pred, y)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    loss = L_task

    return {
        'loss': loss,
        'task': L_task.item(),
    }
