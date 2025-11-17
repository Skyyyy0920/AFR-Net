"""
Routing and load-computation utilities based on an electrical network analogy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .utils import (
    build_incidence_matrix,
    laplacian_from_conductance,
    solve_potentials,
    edge_flows_from_potential,
    standardize
)


def compute_detour_kernel(
    S: torch.Tensor,
    H: int = 5,
    rho: float = 0.8,
) -> torch.Tensor:
    """
    Compute a truncated walk kernel that removes 0/1-hop contributions.

    K = sum_{h=2..H} rho^h * A^h, where A is the row-normalized transition matrix.

    Args:
        S: [N, N] structural connectivity matrix.
        H: Maximum hop count.
        rho: Decay factor for longer walks.

    Returns:
        K: [N, N] detour kernel matrix.
    """
    A = S
    
    # Accumulate powers of A
    K = torch.zeros_like(S)
    A_power = A @ A  # Start from A^2

    for h in range(2, H + 1):
        K = K + (rho ** h) * A_power
        if h < H:
            A_power = A_power @ A
    
    return K


def compute_load(
    S: torch.Tensor,
    F: torch.Tensor,
    H: torch.Tensor,
    edge_gate: nn.Module,
    theta: float = 2.0,
    num_pairs: int = 1024,
    eps: float = 1e-6,
    delta: float = 1e-6,
    detour_H: int = 5,
    detour_rho: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable routing module that computes edge information loads.

    Steps:
        1. Build the edge list from S.
        2. Compute base resistance c_e = -log(S_e + eps).
        3. Obtain gate values a_e from the EdgeGate.
        4. Compute conductance g_e = exp(-c_e + theta * a_e).
        5. Compute the detour kernel K.
        6. Compute pairwise demand M = F * K.
        7. Sample source-target pairs and solve the electrical network.
        8. Aggregate edge loads L.

    Args:
        S: [N, N] structural connectivity.
        F: [N, N] functional connectivity.
        H: [N, d] node embeddings.
        edge_gate: EdgeGate module.
        theta: Gate influence coefficient.
        num_pairs: Number of source-target pairs to sample.
        eps: Numerical stability constant.
        delta: Ridge regularization for the Laplacian.
        detour_H: Maximum hop for the detour kernel.
        detour_rho: Decay factor for the detour kernel.

    Returns:
        L: [E] standardized edge loads.
        edge_index: [2, E] edge indices.
    """
    N = S.shape[0]
    device = S.device
    
    # 1. Build edge list (upper-triangular only)
    from .utils import build_edge_index_from_S
    edge_index, S_e = build_edge_index_from_S(S)  # [2, E], [E]
    E = edge_index.shape[1]
    
    # Gather functional connectivity for each edge
    F_e = F[edge_index[0], edge_index[1]]  # [E]
    
    # 2. Base resistance
    c_e = -torch.log(S_e + eps)  # [E]
    
    # 3. Edge gating
    a_e = edge_gate(H, edge_index, S_e, F_e)  # [E]
    
    # 4. Conductance
    g_e = torch.exp(-c_e + theta * a_e)  # [E]
    
    # 5. Compute detour kernel
    K = compute_detour_kernel(S, H=detour_H, rho=detour_rho)  # [N, N]
    
    # 6. Pairwise demand
    M = F * K  # [N, N]
    
    # 7. Sample pairs
    # Use |M| as sampling weights
    W = M.abs()  # [N, N]
    T = W.sum()  # Scalar
    
    # Avoid degenerate zero-demand case
    if T < eps:
        # If the demand matrix is empty, return zero load
        L = torch.zeros(E, device=device)
        return standardize(L), edge_index
    
    # Build sampling probabilities
    P = W / T  # [N, N]
    
    # Sample with replacement
    num_pairs = min(num_pairs, N * (N - 1) // 2)  # No more than total pairs
    
    # Flatten probabilities for sampling
    P_flat = P.reshape(-1)  # [N*N] - use reshape to avoid non-contiguous tensors
    sampled_flat_idx = torch.multinomial(P_flat, num_pairs, replacement=True)  # [num_pairs]
    
    # Convert to 2D indices
    sampled_i = sampled_flat_idx // N
    sampled_j = sampled_flat_idx % N
    pair_indices = torch.stack([sampled_i, sampled_j], dim=0)  # [2, num_pairs]
    
    # Pair weight alpha = sign(M_ij) * (T / num_pairs)
    M_sampled = M[sampled_i, sampled_j]  # [num_pairs]
    alpha = torch.sign(M_sampled) * (T / num_pairs)  # [num_pairs]
    
    # 8. Electrical network solve
    # Build incidence matrix
    Bmat = build_incidence_matrix(edge_index, N)  # [E, N]
    
    # Build Laplacian
    Lg = laplacian_from_conductance(Bmat, g_e, delta=delta)  # [N, N]
    
    # Solve potentials
    Phi = solve_potentials(Lg, pair_indices, N)  # [N, num_pairs]
    
    # Compute edge flows
    flows = edge_flows_from_potential(Bmat, Phi, g_e, eps=eps)  # [E, num_pairs]
    
    # 9. Aggregate loads
    L_raw = (flows * alpha.unsqueeze(0)).sum(dim=1)  # [E]
    
    # 10. Standardize
    L = standardize(L_raw, eps=eps)  # [E]
    
    return L, edge_index, a_e


def mask_from_L(L: torch.Tensor, tau: float = 8.0, threshold: nn.Parameter = None) -> torch.Tensor:
    """
    Generate a soft mask from the load values.

    m = sigmoid(tau * (L - t))

    Args:
        L: [E] edge loads.
        tau: Temperature parameter.
        threshold: Optional learnable threshold (defaults to 0).

    Returns:
        m: [E] soft mask in [0, 1].
    """
    t = threshold if threshold is not None else 0.0
    m = torch.sigmoid(tau * (L - t))
    return m


def select_subgraph_from_L(
    L: torch.Tensor,
    edge_index: torch.Tensor,
    S: torch.Tensor,
    k: int = None,
    budget_lambda: float = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select a hard subgraph from loads L (used during inference).

    Supported strategies:
        1. Top-k: keep the k edges with highest load.
        2. Budgeted: placeholder for budget-aware selection.

    Args:
        L: [E] edge loads.
        edge_index: [2, E] edge indices.
        S: [N, N] structural connectivity (for optional repairs).
        k: Number of edges to keep (if specified).
        budget_lambda: Budget penalty (if specified).

    Returns:
        edge_index_sub: [2, E_sub] indices of selected edges.
        mask_sub: [E] boolean mask.
    """
    E = L.shape[0]
    
    if k is not None:
        # Top-k strategy
        k = min(k, E)
        _, top_indices = torch.topk(L, k)
        mask_sub = torch.zeros(E, dtype=torch.bool, device=L.device)
        mask_sub[top_indices] = True
    else:
        # Simple threshold strategy (positive loads)
        mask_sub = L > 0
    
    # Extract subgraph edge indices
    edge_index_sub = edge_index[:, mask_sub]
    
    return edge_index_sub, mask_sub

