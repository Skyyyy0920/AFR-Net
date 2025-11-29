"""
Loss functions for the DIAL model, covering both task loss and regularizers.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_losses(
        y_pred: torch.Tensor,
        y: torch.Tensor,
        task: str,
        sparsity_terms=None,
        lambda_sparsity: float = 0.0,
) -> Dict[str, torch.Tensor]:
    if task == 'classification':
        L_task = F.cross_entropy(y_pred, y)
    elif task == 'regression':
        L_task = F.mse_loss(y_pred, y)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    if sparsity_terms is None:
        sparsity_terms = []

    if isinstance(sparsity_terms, torch.Tensor):
        term_list = [sparsity_terms.reshape(-1)]
    else:
        term_list = [t.reshape(-1) for t in sparsity_terms if t is not None and t.numel() > 0]

    if len(term_list) == 0:
        flat_terms = torch.tensor([], device=y_pred.device, dtype=y_pred.dtype)
    else:
        flat_terms = torch.cat(term_list, dim=0)

    if flat_terms.numel() == 0:
        L_sparsity = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
    else:
        L_sparsity = torch.mean(torch.abs(flat_terms))

    total_loss = L_task + lambda_sparsity * L_sparsity

    return {
        'loss': total_loss,
        'sparsity_loss': L_sparsity,
    }
