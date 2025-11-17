"""
Utility helpers for building graph structures, masks, and normalization.
"""

import torch
from typing import Tuple, Optional


def build_edge_index_from_S(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge indices and weights from a structural connectivity matrix.

    Args:
        S: [N, N] symmetric adjacency matrix with non-negative entries.

    Returns:
        edge_index: [2, E] edge indices (upper triangle only).
        edge_weight: [E] edge weights.
    """
    triu_mask = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)
    edges = (S > 0) & triu_mask

    row, col = torch.where(edges)
    edge_index = torch.stack([row, col], dim=0)  # [2, E]

    edge_weight = S[row, col]  # [E]

    return edge_index, edge_weight


def build_incidence_matrix(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    """
    Build the edge-node incidence matrix for an undirected graph.

    Args:
        edge_index: [2, E] edge indices.
        N: Number of nodes.

    Returns:
        Bmat: [E, N] incidence matrix with +1 at the source and -1 at the target.
    """
    E = edge_index.shape[1]
    device = edge_index.device

    Bmat = torch.zeros(E, N, device=device, dtype=torch.float32)
    edge_idx = torch.arange(E, device=device)
    Bmat[edge_idx, edge_index[0]] = 1.0
    Bmat[edge_idx, edge_index[1]] = -1.0

    return Bmat


def laplacian_from_conductance(
    Bmat: torch.Tensor,
    g_e: torch.Tensor,
    delta: float = 1e-6
) -> torch.Tensor:
    """
    Build the graph Laplacian from edge conductances with ridge regularization.

    Args:
        Bmat: [E, N] incidence matrix.
        g_e: [E] edge conductances.
        delta: Ridge coefficient added to the diagonal.

    Returns:
        Lg: [N, N] Laplacian matrix (regularized to stay invertible).
    """
    G_diag = torch.diag(g_e)
    Lg = Bmat.t() @ G_diag @ Bmat

    N = Lg.shape[0]
    Lg = Lg + delta * torch.eye(N, device=Lg.device, dtype=Lg.dtype)

    return Lg


def solve_potentials(
    Lg: torch.Tensor,
    pair_indices: torch.Tensor,
    N: int
) -> torch.Tensor:
    """
    Solve the potentials for multiple source-target pairs.

    Args:
        Lg: [N, N] Laplacian matrix.
        pair_indices: [2, B] source-target node indices.
        N: Number of nodes.

    Returns:
        Phi: [N, B] potentials per pair.
    """
    B = pair_indices.shape[1]
    device = Lg.device

    RHS = torch.zeros(N, B, device=device, dtype=Lg.dtype)

    batch_idx = torch.arange(B, device=device)
    RHS[pair_indices[0], batch_idx] = 1.0  # Source +1
    RHS[pair_indices[1], batch_idx] = -1.0  # Sink -1

    try:
        Phi = torch.linalg.solve(Lg, RHS)
    except RuntimeError as e:
        print(f"Warning: Laplacian solve failed, falling back to least squares: {e}")
        Phi = torch.linalg.lstsq(Lg, RHS).solution

    return Phi


def edge_flows_from_potential(
    Bmat: torch.Tensor,
    Phi: torch.Tensor,
    g_e: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Convert node potentials into edge flows (basis for information load).

    Args:
        Bmat: [E, N] incidence matrix.
        Phi: [N, B] potentials.
        g_e: [E] edge conductances.
        eps: Numerical stability constant.

    Returns:
        flows: [E, B] flow magnitude per edge and per pair.
    """
    Delta = Bmat @ Phi

    flows = g_e.unsqueeze(1) * torch.sqrt(Delta ** 2 + eps ** 2)

    return flows


def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalize a tensor.

    Args:
        x: Input tensor.
        eps: Numerical stability constant.

    Returns:
        Standardized tensor.
    """
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def create_attention_mask_from_adjacency(
    S: torch.Tensor,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Create an attention mask from an adjacency matrix.

    Args:
        S: [N, N] adjacency matrix.
        add_self_loops: Whether to include self-loops.

    Returns:
        mask: [N, N] attention mask (-inf for non-adjacent pairs).
    """
    N = S.shape[0]
    device = S.device

    adj_mask = (S > 0).float()

    if add_self_loops:
        adj_mask = adj_mask + torch.eye(N, device=device, dtype=adj_mask.dtype)

    attention_mask = torch.where(
        adj_mask > 0,
        torch.zeros_like(adj_mask),
        torch.full_like(adj_mask, float('-inf'))
    )

    return attention_mask


def select_top_k_edges(
    L: torch.Tensor,
    k: int,
    ensure_connected: bool = True,
    S: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Select the top-k edges (optionally fixing connectivity).

    Args:
        L: [E] edge load/importance scores.
        k: Number of edges to keep.
        ensure_connected: Flag placeholder for future connectivity repairs.
        S: [N, N] structural connectivity matrix.
        edge_index: [2, E] edge indices.

    Returns:
        Boolean mask of shape [E] indicating selected edges.
    """
    E = L.shape[0]
    k = min(k, E)

    _, top_indices = torch.topk(L, k)
    mask = torch.zeros(E, dtype=torch.bool, device=L.device)
    mask[top_indices] = True

    # TODO: add connectivity repair if needed.

    return mask
