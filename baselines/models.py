"""
Baseline models for FC/SC inputs with dual-stream encoders.
Each baseline encodes FC and SC separately, concatenates embeddings,
then feeds them to a shared prediction head.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, global_mean_pool

from dial.model import GraphormerNodeEncoder


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
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, embed_dim, heads=1, dropout=dropout, concat=True)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        # Do not pass edge_weight to GAT to avoid overriding attention.
        h = self.conv1(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.act(h)
        h = global_mean_pool(h, batch_vec)
        return h  # [B, embed_dim]


class GATv2Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, embed_dim, heads=1, dropout=dropout, concat=True)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        # Do not pass edge_weight to GATv2 to keep attention weights learnable.
        h = self.conv1(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
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


class _GraphConvDualStream(nn.Module):
    """
    Shared dual-stream wrapper for GCN/GAT/GATv2 style encoders that take (x, edge_index, batch_vec).
    """

    def __init__(self, encoder_ctor, num_nodes: int, hidden_dim: int, embed_dim: int, num_classes: int, dropout: float, heads: int = 4):
        super().__init__()
        if encoder_ctor in (GATEncoder, GATv2Encoder):
            self.fc_encoder = encoder_ctor(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, heads=heads, dropout=dropout)
            self.sc_encoder = encoder_ctor(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, heads=heads, dropout=dropout)
            head_in_dim = embed_dim * 2
            head_hidden = hidden_dim * heads
        else:
            self.fc_encoder = encoder_ctor(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
            self.sc_encoder = encoder_ctor(num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout)
            head_in_dim = embed_dim * 2
            head_hidden = hidden_dim * 2
        self.head = BaselineHead(head_in_dim, hidden_dim=head_hidden, num_classes=num_classes, dropout=dropout)

    def _encode_single(self, adj: torch.Tensor, encoder: nn.Module, use_edge_weight: bool) -> torch.Tensor:
        """
        adj: [N, N] adjacency for one graph
        """
        N = adj.shape[0]
        edge_index, edge_weight = build_edge_index_from_adj(adj)
        x = adj  # use adjacency rows as node features
        batch_vec = torch.zeros(N, dtype=torch.long, device=adj.device)
        if use_edge_weight:
            return encoder(x, edge_index, edge_weight, batch_vec).squeeze(0)
        return encoder(x, edge_index, batch_vec).squeeze(0)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GCNBaseline(_GraphConvDualStream):
    """
    FC and SC each go through a two-layer GCN; embeddings are concatenated and fed to a head.
    Node features are adjacency rows.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(GCNEncoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        if fc_adj.dim() == 2:
            fc_adj = fc_adj.unsqueeze(0)
        if sc_adj.dim() == 2:
            sc_adj = sc_adj.unsqueeze(0)

        fc_emb = []
        sc_emb = []
        for b in range(fc_adj.shape[0]):
            fc_emb.append(self._encode_single(fc_adj[b], self.fc_encoder, use_edge_weight=True))
            sc_emb.append(self._encode_single(sc_adj[b], self.sc_encoder, use_edge_weight=True))

        feat = torch.cat([torch.stack(fc_emb, dim=0), torch.stack(sc_emb, dim=0)], dim=-1)
        return self.head(feat)


class GATBaseline(_GraphConvDualStream):
    """
    FC and SC each go through a two-layer GAT; embeddings are concatenated and fed to a head.
    Node features are adjacency rows. edge_weight is intentionally not used.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(GATEncoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout, heads=heads)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        if fc_adj.dim() == 2:
            fc_adj = fc_adj.unsqueeze(0)
        if sc_adj.dim() == 2:
            sc_adj = sc_adj.unsqueeze(0)

        fc_emb = []
        sc_emb = []
        for b in range(fc_adj.shape[0]):
            fc_emb.append(self._encode_single(fc_adj[b], self.fc_encoder, use_edge_weight=False))
            sc_emb.append(self._encode_single(sc_adj[b], self.sc_encoder, use_edge_weight=False))

        feat = torch.cat([torch.stack(fc_emb, dim=0), torch.stack(sc_emb, dim=0)], dim=-1)
        return self.head(feat)


class GATv2Baseline(_GraphConvDualStream):
    """
    FC and SC each go through a two-layer GATv2; embeddings are concatenated and fed to a head.
    Node features are adjacency rows. edge_weight is intentionally not used.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(GATv2Encoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout, heads=heads)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        if fc_adj.dim() == 2:
            fc_adj = fc_adj.unsqueeze(0)
        if sc_adj.dim() == 2:
            sc_adj = sc_adj.unsqueeze(0)

        fc_emb = []
        sc_emb = []
        for b in range(fc_adj.shape[0]):
            fc_emb.append(self._encode_single(fc_adj[b], self.fc_encoder, use_edge_weight=False))
            sc_emb.append(self._encode_single(sc_adj[b], self.sc_encoder, use_edge_weight=False))

        feat = torch.cat([torch.stack(fc_emb, dim=0), torch.stack(sc_emb, dim=0)], dim=-1)
        return self.head(feat)


class GraphormerBaseline(nn.Module):
    """
    Dual-stream Graphormer baseline. Each stream uses its own GraphormerNodeEncoder.
    """

    expects_graphormer_inputs = True

    def __init__(
            self,
            num_nodes: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            num_classes: int = 2,
            dropout: float = 0.1,
            max_degree: int = 511,
            max_path_len: int = 5,
    ):
        super().__init__()
        encoder_kwargs = dict(
            N=num_nodes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )
        self.fc_encoder = GraphormerNodeEncoder(**encoder_kwargs)
        self.sc_encoder = GraphormerNodeEncoder(**encoder_kwargs)
        self.head = BaselineHead(
            in_dim=d_model * 2,
            hidden_dim=dim_feedforward // 2,
            num_classes=num_classes,
            dropout=dropout
        )
        self.graph_dropout = nn.Dropout(dropout * 0.5)

    @staticmethod
    def _pool_nodes(node_repr: torch.Tensor) -> torch.Tensor:
        # Mean pooling over nodes for graph embedding.
        return node_repr.mean(dim=1)

    def forward(
            self,
            fc_node_feat: torch.Tensor,
            fc_in_degree: torch.Tensor,
            fc_out_degree: torch.Tensor,
            fc_path_data: torch.Tensor,
            fc_dist: torch.Tensor,
            fc_attn_mask: torch.Tensor,
            sc_node_feat: torch.Tensor,
            sc_in_degree: torch.Tensor,
            sc_out_degree: torch.Tensor,
            sc_path_data: torch.Tensor,
            sc_dist: torch.Tensor,
            sc_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        fc_repr = self.fc_encoder(
            fc_node_feat, fc_in_degree, fc_out_degree, fc_path_data, fc_dist, fc_attn_mask
        )
        sc_repr = self.sc_encoder(
            sc_node_feat, sc_in_degree, sc_out_degree, sc_path_data, sc_dist, sc_attn_mask
        )

        fc_graph = self.graph_dropout(self._pool_nodes(fc_repr))
        sc_graph = self.graph_dropout(self._pool_nodes(sc_repr))
        feat = torch.cat([fc_graph, sc_graph], dim=-1)
        return self.head(feat)
