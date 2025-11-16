import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List

from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder

from scipy.sparse.csgraph import shortest_path as scipy_shortest_path

from .routing import compute_load, mask_from_L, select_subgraph_from_L
from .losses import compute_losses
from .utils import build_edge_index_from_S, create_attention_mask_from_adjacency


# ============================================================================
# Node Encoder - Graphormer
# ============================================================================

class GraphormerNodeEncoder(nn.Module):
    """Graphormer node encoder implemented with DGL GraphormerLayer."""

    def __init__(
            self,
            N: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            max_spd: int = 512,
            max_degree: int = 200,
            max_path_len: int = 5,
            edge_feat_dim: int = 1,
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.max_spd = max_spd
        self.max_degree = max_degree
        self.max_path_len = max(1, max_path_len)
        self.edge_feat_dim = max(1, edge_feat_dim)

        self.fc_feature_proj = nn.Sequential(
            nn.Linear(N, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=d_model)
        self.spatial_encoder = SpatialEncoder(max_dist=max_spd, num_heads=nhead)
        self.path_encoder = PathEncoder(max_len=self.max_path_len, feat_dim=self.edge_feat_dim, num_heads=nhead)

        self.layers = nn.ModuleList([
            GraphormerLayer(
                feat_size=d_model,
                hidden_size=dim_feedforward,
                num_heads=nhead,
                dropout=dropout,
                activation=nn.GELU(),
                norm_first=False
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.fc_feature_proj(node_feat)
        degree_emb = self.degree_encoder(torch.stack((in_degree, out_degree), dim=0))
        H = x + degree_emb

        dist_long = dist.long()
        path_bias = self.path_encoder(dist_long, path_data)
        spatial_bias = self.spatial_encoder(dist_long)
        attn_bias = path_bias + spatial_bias

        for layer in self.layers:
            H = layer(H, attn_mask=attn_mask, attn_bias=attn_bias)

        return self.norm(H)


class EdgeGate(nn.Module):
    """边门控：为每条边生成门值 a_e ∈ [0, 1]"""

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        edge_feature_dim = 2 * d_model + 2

        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
            self,
            H: torch.Tensor,
            edge_index: torch.Tensor,
            S_e: torch.Tensor,
            F_e: torch.Tensor
    ) -> torch.Tensor:
        H_u = H[edge_index[0]]
        H_v = H[edge_index[1]]
        H_diff = torch.abs(H_u - H_v)
        H_prod = H_u * H_v

        edge_features = torch.cat([
            H_diff,
            H_prod,
            S_e.unsqueeze(1),
            F_e.unsqueeze(1)
        ], dim=1)

        a_e = self.mlp(edge_features).squeeze(1)
        return a_e


# ============================================================================
# 掩码图Transformer
# ============================================================================

class MaskedGraphTransformer(nn.Module):
    """掩码图Transformer：在软掩码的子图上做表征学习"""

    def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            H: torch.Tensor,
            edge_index: torch.Tensor,
            m: torch.Tensor,
            eps: float = 1e-9
    ) -> torch.Tensor:
        N = H.shape[0]
        device = H.device

        attn_bias = self._build_attention_bias(N, edge_index, m, eps, device)

        Z = H.unsqueeze(0)  # [1, N, d_model]

        Z = self.transformer(Z, mask=attn_bias)

        Z = Z.squeeze(0)
        Z = self.norm(Z)

        return Z

    def _build_attention_bias(
            self,
            N: int,
            edge_index: torch.Tensor,
            m: torch.Tensor,
            eps: float,
            device: torch.device
    ) -> torch.Tensor:
        # 初始化为-inf（非边对）
        attn_bias = torch.full((N, N), float('-inf'), device=device)
        attn_bias.fill_diagonal_(0.0)  # 自环

        # 为边对设置偏置：log(m_e + eps)
        row, col = edge_index[0], edge_index[1]
        bias_values = torch.log(m + eps)
        attn_bias[row, col] = bias_values
        attn_bias[col, row] = bias_values  # 对称

        return attn_bias


# ============================================================================
# 预测头
# ============================================================================

class AttentionPooling(nn.Module):
    """注意力池化"""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention(Z)
        attn_weights = F.softmax(attn_scores, dim=0)
        pooled = (Z * attn_weights).sum(dim=0)
        return pooled


class PredictionHead(nn.Module):
    """任务预测头"""

    def __init__(
            self,
            d_model: int = 64,
            num_classes: int = 2,
            task: str = 'classification',
            pooling: str = 'mean',
            hidden_dim: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.task = task
        self.pooling = pooling

        if pooling == 'attention':
            self.attention_pooling = AttentionPooling(d_model)

        if task == 'classification':
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task == 'regression':
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"不支持的任务类型: {task}")

    def forward(
            self,
            Z: torch.Tensor,
            m: torch.Tensor = None,
            edge_index: torch.Tensor = None
    ) -> torch.Tensor:
        if self.pooling == 'mean':
            graph_repr = Z.mean(dim=0)
        elif self.pooling == 'attention':
            graph_repr = self.attention_pooling(Z)
        else:
            raise ValueError(f"不支持的池化方式: {self.pooling}")

        y_pred = self.mlp(graph_repr)

        if self.task == 'regression':
            y_pred = y_pred.squeeze(-1)

        return y_pred


# ============================================================================
# 顶层模型
# ============================================================================

class DIALModel(nn.Module):
    """
    Flow→Load→Mask→Subgraph 端到端模型
    
    流程:
    1. GraphormerNodeEncoder: S,F → H (节点嵌入)
    2. 可微软路由: S,F,H → L (信息载荷)
    3. 软掩码: L → m
    4. MaskedGraphTransformer: H,m → Z (掩码子图表征)
    5. PredictionHead: Z → y_pred (任务预测)
    6. 损失计算: 任务损失 + 正则化
    """

    def __init__(
            self,
            N: int,
            d_model: int = 64,
            nhead: int = 4,
            num_node_layers: int = 2,
            num_graph_layers: int = 2,
            dim_feedforward: int = 256,
            num_classes: int = 2,
            task: str = 'classification',
            dropout: float = 0.1,
            # 路由参数
            theta: float = 2.0,
            num_pairs: int = 1024,
            detour_H: int = 5,
            detour_rho: float = 0.6,
            # 掩码参数
            tau: float = 8.0,
            # 损失系数
            lambda_align: float = 0.2,
            lambda_budget: float = 0.05,
            lambda_gate: float = 1e-4,
            # 数值稳定性
            eps: float = 1e-6,
            delta: float = 1e-6
    ):
        super().__init__()

        self.N = N
        self.d_model = d_model
        self.task = task

        # 超参数
        self.theta = theta
        self.num_pairs = num_pairs
        self.detour_H = detour_H
        self.detour_rho = detour_rho
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_budget = lambda_budget
        self.lambda_gate = lambda_gate
        self.eps = eps
        self.delta = delta

        self.node_encoder = GraphormerNodeEncoder(
            N=N,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_node_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.edge_gate = EdgeGate(
            d_model=d_model,
            hidden_dim=dim_feedforward // 2
        )

        self.threshold = nn.Parameter(torch.tensor(0.0))

        self.graph_transformer = MaskedGraphTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_graph_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.prediction_head = PredictionHead(
            d_model=d_model,
            num_classes=num_classes,
            task=task,
            pooling='mean',
            hidden_dim=dim_feedforward // 2
        )

    def _split_batch_inputs(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            S: torch.Tensor,
            F: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        batch_size = node_feat.shape[0]
        entries: List[Dict[str, torch.Tensor]] = []
        for idx in range(batch_size):
            entry = {
                'node_feat': node_feat[idx:idx + 1],
                'in_degree': in_degree[idx:idx + 1],
                'out_degree': out_degree[idx:idx + 1],
                'path_data': path_data[idx:idx + 1],
                'dist': dist[idx:idx + 1],
                'attn_mask': attn_mask[idx:idx + 1] if attn_mask is not None else None,
                'S': S[idx],
                'F': F[idx]
            }
            entries.append(entry)
        return entries

    def forward(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            S: torch.Tensor,
            F: torch.Tensor,
            y: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        entries = self._split_batch_inputs(
            node_feat, in_degree, out_degree, path_data, dist, attn_mask, S, F
        )
        targets = y.squeeze(-1) if y is not None and y.dim() > 1 else y

        y_pred_list = []
        losses = []
        diag_list = []

        for idx, entry in enumerate(entries):
            node_feat = entry['node_feat'] if entry['node_feat'] is not None else entry['F'].unsqueeze(0)
            in_degree = entry['in_degree']
            out_degree = entry['out_degree']
            path_data = entry['path_data']
            dist = entry['dist']
            attn_mask = entry['attn_mask']

            H = self.node_encoder(
                node_feat, in_degree, out_degree, path_data, dist, attn_mask
            ).squeeze(0)

            L, edge_index = compute_load(
                entry['S'], entry['F'], H,
                edge_gate=self.edge_gate,
                theta=self.theta,
                num_pairs=self.num_pairs,
                eps=self.eps,
                delta=self.delta,
                detour_H=self.detour_H,
                detour_rho=self.detour_rho
            )

            m = mask_from_L(L, tau=self.tau, threshold=self.threshold)
            Z = self.graph_transformer(H, edge_index, m, eps=self.eps)
            y_pred = self.prediction_head(Z, m, edge_index)

            loss = None
            loss_dict = {}
            if targets is not None:
                target = targets[idx] if targets.dim() > 0 else targets
                _, S_e = build_edge_index_from_S(entry['S'])
                a_e = self.edge_gate.last_gate_values
                loss_dict = compute_losses(
                    y_pred=y_pred,
                    y=target,
                    L=L,
                    m=m,
                    task=self.task,
                    S_e=S_e,
                    a_e=a_e,
                    lambda_align=self.lambda_align,
                    lambda_budget=self.lambda_budget,
                    lambda_gate=self.lambda_gate,
                    eps=self.eps
                )
                loss = loss_dict['loss']
                losses.append(loss)

            diag = {
                'L': L.detach(),
                'm': m.detach(),
                'edge_index': edge_index.detach(),
                'H_node': H.detach(),
                'Z_graph': Z.detach(),
                'y_pred': y_pred.detach(),
                'loss_dict': loss_dict
            }

            y_pred_list.append(y_pred.unsqueeze(0))
            diag_list.append(diag)

        y_pred_batch = torch.cat(y_pred_list, dim=0)
        avg_loss = torch.stack(losses).mean() if losses else None
        batch_diag = {'batch_diag': diag_list}
        return y_pred_batch, avg_loss, batch_diag

    def inference(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            S: torch.Tensor,
            F: torch.Tensor,
            k: Optional[int] = None,
            budget_lambda: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """推理模式：使用硬子图选择"""
        self.eval()

        with torch.no_grad():
            entries = self._split_batch_inputs(
                node_feat, in_degree, out_degree, path_data, dist, attn_mask, S, F
            )

            preds = []
            edge_indices = []
            masks = []

            for entry in entries:
                H = self.node_encoder(
                    entry['node_feat'] if entry['node_feat'] is not None else entry['F'].unsqueeze(0),
                    entry['in_degree'],
                    entry['out_degree'],
                    entry['path_data'],
                    entry['dist'],
                    entry['attn_mask']
                ).squeeze(0)

                L, edge_index = compute_load(
                    entry['S'], entry['F'], H,
                    edge_gate=self.edge_gate,
                    theta=self.theta,
                    num_pairs=self.num_pairs,
                    eps=self.eps,
                    delta=self.delta,
                    detour_H=self.detour_H,
                    detour_rho=self.detour_rho
                )

                edge_index_sub, mask_sub = select_subgraph_from_L(
                    L, edge_index, entry['S'],
                    k=k,
                    budget_lambda=budget_lambda
                )

                m_hard = mask_sub.float()
                Z = self.graph_transformer(H, edge_index, m_hard, eps=self.eps)
                y_pred = self.prediction_head(Z, m_hard, edge_index)

                preds.append(y_pred.unsqueeze(0))
                edge_indices.append(edge_index_sub)
                masks.append(mask_sub)

        return torch.cat(preds, dim=0), edge_indices, masks

    def get_subgraph_importance(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            S: torch.Tensor,
            F: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取边的重要性分数"""
        self.eval()

        with torch.no_grad():
            entries = self._split_batch_inputs(
                node_feat, in_degree, out_degree, path_data, dist, attn_mask, S, F
            )
            loads = []
            edges = []

            for entry in entries:
                H = self.node_encoder(
                    entry['node_feat'] if entry['node_feat'] is not None else entry['F'].unsqueeze(0),
                    entry['in_degree'],
                    entry['out_degree'],
                    entry['path_data'],
                    entry['dist'],
                    entry['attn_mask']
                ).squeeze(0)

                L, edge_index = compute_load(
                    entry['S'], entry['F'], H,
                    edge_gate=self.edge_gate,
                    theta=self.theta,
                    num_pairs=self.num_pairs,
                    eps=self.eps,
                    delta=self.delta,
                    detour_H=self.detour_H,
                    detour_rho=self.detour_rho
                )
                loads.append(L)
                edges.append(edge_index)

            if len(loads) == 1:
                return loads[0], edges[0]
            return loads, edges
