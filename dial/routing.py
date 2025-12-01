"""
Routing and energy-computation utilities using closed-form energy (Route A).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .utils import (
    build_incidence_matrix,
    laplacian_from_conductance,
    standardize,
    build_edge_index_from_S,
    build_task_laplacian,
)


def _safe_solve(mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    Solve a linear system with a fallback to least squares for stability.
    """
    try:
        return torch.linalg.solve(mat, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(mat, rhs).solution


def compute_edge_energy(
        S: torch.Tensor,
        F: torch.Tensor,
        H: torch.Tensor,
        edge_gate: nn.Module,
        delta: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-edge energy E_e using Route B (low-rank / eigen approximation).

    Args:
        S: [N, N] structural connectivity.
        F: [N, N] functional/feature similarity (used as task matrix T).
        H: [N, d_model] node embeddings.
        edge_gate: Module producing edge conductances.
        delta: Ridge for Laplacian.

    Returns:
        E_e: [E] energy per edge.
        edge_index: [2, E] edge indices.
        S_e: [E] structural weights per edge.
        a_e: [E] pre-activation edge scores.
    """
    N = S.shape[0]
    device = S.device

    edge_index, S_e = build_edge_index_from_S(S)  # [2, E], [E]
    E = edge_index.shape[1]

    if E == 0:
        empty = torch.zeros(0, device=device, dtype=H.dtype)
        return empty, edge_index, S_e, empty

    a_e = edge_gate(H, edge_index)  # [E]
    g_e = torch.exp(a_e)
    g_e = torch.clamp(g_e, min=1e-6, max=1e6)  # conductance, positive and bounded

    Bmat = build_incidence_matrix(edge_index, N)  # [E, N]
    Lg = laplacian_from_conductance(Bmat, g_e, delta=delta)  # [N, N]
    L_T = build_task_laplacian(F)  # [N, N]

    # Route B: eigen-decomposition of task Laplacian
    vals, vecs = torch.linalg.eigh(L_T)  # vals: [N], vecs: [N, N]

    # Solve Lg * Y = vecs (all eigenvectors in one shot)
    Y = _safe_solve(Lg, vecs)  # [N, N]

    # Vectorized energy: 2 g_uv * sum_k vals_k (Y_u - Y_v)^2
    diff = Y[edge_index[0]] - Y[edge_index[1]]  # [E, N]
    energy_core = (diff * diff) * vals.unsqueeze(0)  # [E, N]
    edge_energy = 2.0 * g_e * energy_core.sum(dim=1)  # [E]

    return edge_energy, edge_index, S_e, a_e


def mask_from_energy(
        E: torch.Tensor,
        S_e: torch.Tensor,
        tau: float,
        lambda_cost: float,
        threshold: nn.Parameter,
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Build a differentiable edge mask from energies.

    mask_e = sigmoid( tau * ( zscore(E_e) - lambda_cost * c_e - threshold ) )
    where c_e = 1 / (S_e + eps).
    """
    if E.numel() == 0:
        return E

    E_norm = standardize(E, eps=eps)
    c_e = 1.0 / (S_e + eps)
    logits = tau * (E_norm - lambda_cost * c_e - threshold)
    return torch.sigmoid(logits)


def get_ste_mask(
        energies: torch.Tensor,
        tau: float = 1.0,
        k: Optional[int] = None,
        threshold: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Straight-through estimator for hard edge masks.

    Forward: build a hard binary mask using Top-K or thresholding on energies.
    Backward: use sigmoid-soft probabilities for gradient flow.
    """
    if energies.numel() == 0:
        empty = torch.zeros_like(energies)
        return empty, empty

    log_E = torch.log(energies + 1e-9)
    logits = (log_E - log_E.mean()) / max(tau, 1e-6)
    m_soft = torch.sigmoid(logits)

    num_edges = energies.shape[0]
    if k is not None:
        k = max(0, min(k, num_edges))
        if k == 0:
            m_hard = torch.zeros_like(m_soft)
        else:
            _, idx = torch.topk(energies, k)
            m_hard = torch.zeros_like(m_soft)
            m_hard[idx] = 1.0
    elif threshold is not None:
        m_hard = (energies > threshold).float()
    else:
        # Default to stochastic gate driven by soft mask
        m_hard = (m_soft >= 0.5).float()

    mask = (m_hard - m_soft).detach() + m_soft
    return mask, m_soft


def select_subgraph_from_energy(
        E: torch.Tensor,
        edge_index: torch.Tensor,
        k: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select a hard subgraph using edge energies (used optionally at inference).
    """
    num_edges = E.shape[0]
    if num_edges == 0:
        return edge_index, torch.zeros(0, dtype=torch.bool, device=E.device)

    if k is not None:
        k = min(k, num_edges)
        _, idx = torch.topk(E, k)
        mask = torch.zeros(num_edges, dtype=torch.bool, device=E.device)
        mask[idx] = True
    else:
        mask = E > 0

    return edge_index[:, mask], mask
