"""
Baseline models for FC/SC inputs: MLP, GCN, and GAT variants.
Each baseline encodes FC and SC separately, concatenates embeddings,
then feeds them to a shared prediction head.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from typing import Tuple


def build_edge_index_from_adj(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge index and weights from an adjacency matrix (upper triangle to avoid duplicates).
    """
    device = adj.device
    N = adj.shape[0]
    triu_mask = torch.triu(torch.ones_like(adj, dtype=torch.bool), diagonal=1)
    rows, cols = torch.where((adj != 0) & triu_mask)

    if rows.numel() == 0:
        # Fallback to self-loops to keep graph non-empty
        idx = torch.arange(N, device=device)
        edge_index = torch.stack([idx, idx], dim=0)
        edge_weight = torch.ones(N, device=device, dtype=adj.dtype)
    else:
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = adj[rows, cols]
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    return edge_index, edge_weight


class MLPEncoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = num_nodes * num_nodes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        # adj: [B, N, N] or [N, N]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        B = adj.shape[0]
        x = adj.reshape(B, -1)
        return self.net(x)  # [B, embed_dim]


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = global_mean_pool(h, batch_vec)
        return h  # [B, embed_dim]


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, embed_dim, heads=1, dropout=dropout, concat=True)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = global_mean_pool(h, batch_vec)
        return h  # [B, embed_dim]


class BaselineHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBaseline(nn.Module):
    """
    FC and SC each go through an MLP encoder, then embeddings are concatenated and fed to a head.
    """

    def __init__(self, num_nodes: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc_encoder = MLPEncoder(num_nodes, hidden_dim, embed_dim, dropout)
        self.sc_encoder = MLPEncoder(num_nodes, hidden_dim, embed_dim, dropout)
        self.head = BaselineHead(embed_dim * 2, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        fc_emb = self.fc_encoder(fc_adj)
        sc_emb = self.sc_encoder(sc_adj)
        feat = torch.cat([fc_emb, sc_emb], dim=-1)
        return self.head(feat)


class GCNBaseline(nn.Module):
    """
    FC and SC each go through a two-layer GCN; graph embeddings are concatenated and fed to a head.
    Node features are adjacency rows.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc_encoder = GCNEncoder(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
        self.sc_encoder = GCNEncoder(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
        self.head = BaselineHead(embed_dim * 2, hidden_dim=hidden_dim * 2, num_classes=num_classes, dropout=dropout)

    def _encode_single(self, adj: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        adj: [N, N] adjacency for one graph
        """
        N = adj.shape[0]
        edge_index, edge_weight = build_edge_index_from_adj(adj)
        x = adj  # use adjacency rows as node features
        batch_vec = torch.zeros(N, dtype=torch.long, device=adj.device)
        # encoder returns shape [1, embed_dim] because batch_vec has a single graph; squeeze to [embed_dim]
        return encoder(x, edge_index, edge_weight, batch_vec).squeeze(0)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        if fc_adj.dim() == 2:
            fc_adj = fc_adj.unsqueeze(0)
        if sc_adj.dim() == 2:
            sc_adj = sc_adj.unsqueeze(0)

        B = fc_adj.shape[0]
        fc_emb_list = []
        sc_emb_list = []
        for b in range(B):
            fc_emb_list.append(self._encode_single(fc_adj[b], self.fc_encoder))
            sc_emb_list.append(self._encode_single(sc_adj[b], self.sc_encoder))

        fc_emb = torch.stack(fc_emb_list, dim=0)
        sc_emb = torch.stack(sc_emb_list, dim=0)
        feat = torch.cat([fc_emb, sc_emb], dim=-1)
        return self.head(feat)


class GATBaseline(nn.Module):
    """
    FC and SC each go through a two-layer GAT; graph embeddings are concatenated and fed to a head.
    Node features are adjacency rows.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc_encoder = GATEncoder(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, heads=heads, dropout=dropout)
        self.sc_encoder = GATEncoder(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, heads=heads, dropout=dropout)
        self.head = BaselineHead(embed_dim * 2, hidden_dim=hidden_dim * heads, num_classes=num_classes, dropout=dropout)

    def _encode_single(self, adj: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        N = adj.shape[0]
        edge_index, edge_weight = build_edge_index_from_adj(adj)
        x = adj  # adjacency rows as node features
        batch_vec = torch.zeros(N, dtype=torch.long, device=adj.device)
        # encoder returns shape [1, embed_dim] because batch_vec has a single graph; squeeze to [embed_dim]
        return encoder(x, edge_index, edge_weight, batch_vec).squeeze(0)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        if fc_adj.dim() == 2:
            fc_adj = fc_adj.unsqueeze(0)
        if sc_adj.dim() == 2:
            sc_adj = sc_adj.unsqueeze(0)

        B = fc_adj.shape[0]
        fc_emb_list = []
        sc_emb_list = []
        for b in range(B):
            fc_emb_list.append(self._encode_single(fc_adj[b], self.fc_encoder))
            sc_emb_list.append(self._encode_single(sc_adj[b], self.sc_encoder))

        fc_emb = torch.stack(fc_emb_list, dim=0)
        sc_emb = torch.stack(sc_emb_list, dim=0)
        feat = torch.cat([fc_emb, sc_emb], dim=-1)
        return self.head(feat)
