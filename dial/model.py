import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

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

    def _compute_shortest_path_distance(self, S: torch.Tensor) -> torch.Tensor:
        N = S.shape[0]
        device = S.device
        adj = (S > 0).to(torch.float32)
        adj_np = adj.cpu().numpy()
        dist_np = None
        if scipy_shortest_path is not None:
            try:
                dist_np = scipy_shortest_path(adj_np, directed=False, unweighted=True)
            except Exception as exc:
                print(f"Warning: scipy shortest_path failed ({exc}); falling back to numpy Floyd-Warshall.")
        if dist_np is None:
            dist_np = np.full((N, N), np.inf, dtype=np.float32)
            np.fill_diagonal(dist_np, 0.0)
            dist_np[adj_np > 0] = 1.0
            for k in range(N):
                dist_np = np.minimum(dist_np, dist_np[:, [k]] + dist_np[[k], :])

        dist_np[np.isinf(dist_np)] = self.max_spd
        dist_np = np.clip(dist_np, 0, self.max_spd)
        return torch.from_numpy(dist_np).long().to(device)

    def _compute_degree_centrality(self, S: torch.Tensor) -> torch.Tensor:
        degree = (S > 0).sum(dim=1).long()
        return torch.clamp(degree, 0, self.max_degree - 1)

    def _build_path_data(self, S: torch.Tensor) -> torch.Tensor:
        device = S.device
        N = S.shape[0]
        path_data = torch.zeros(N, N, self.max_path_len, self.edge_feat_dim, device=device)
        edge_mask = S > 0
        if edge_mask.any():
            edge_values = S[edge_mask].float()
            max_val = edge_values.max()
            normalized = torch.zeros_like(S, dtype=path_data.dtype)
            if max_val > 0:
                normalized[edge_mask] = edge_values / (max_val + 1e-8)
            path_data[:, :, 0, 0] = normalized
        return path_data

    def _prepare_degree_embeddings(self, degree: torch.Tensor) -> torch.Tensor:
        deg = degree.unsqueeze(0)
        stacked = torch.stack((deg, deg), dim=0)
        return self.degree_encoder(stacked)

    def _compute_attention_bias(self, dist: torch.Tensor, path_data: torch.Tensor) -> torch.Tensor:
        dist_batch = dist.unsqueeze(0).long()
        path_batch = path_data.unsqueeze(0).float()
        path_bias = self.path_encoder(dist_batch, path_batch)
        spatial_bias = self.spatial_encoder(dist_batch)
        return path_bias + spatial_bias

    def forward(self, S: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        if S.shape[0] != self.N:
            raise ValueError(f"Expected {self.N} nodes, but got {S.shape[0]}")

        degree = self._compute_degree_centrality(S)
        degree_emb = self._prepare_degree_embeddings(degree)
        x = F + degree_emb

        dist = self._compute_shortest_path_distance(S)
        path_data = self._build_path_data(S)
        attn_bias = self._compute_attention_bias(dist, path_data)

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)

        x = self.norm(x.squeeze(0))
        return x


class EdgeGate(nn.Module):
    """边门控：为每条边生成门值 a_e ∈ [0, 1]"""

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        edge_feature_dim = 2 * d_model + 2

        # MLP: 3层，使用SiLU激活
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.last_gate_values = None

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
        self.last_gate_values = a_e
        return a_e

    def get_gate_regularization(self) -> torch.Tensor:
        if self.last_gate_values is None:
            return torch.tensor(0.0)
        return self.last_gate_values.abs().mean()


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

        # 使用PyTorch标准TransformerEncoderLayer
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

        # 构造注意力偏置矩阵
        attn_bias = self._build_attention_bias(N, edge_index, m, eps, device)

        # 添加batch维度
        Z = H.unsqueeze(0)  # [1, N, d_model]

        # Transformer编码（PyTorch使用mask参数）
        Z = self.transformer(Z, mask=attn_bias)

        # 去掉batch维度
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

class FlowLoadMaskModel(nn.Module):
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

        # 组件
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

    def forward(
            self,
            S: torch.Tensor,
            F: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            task: Optional[str] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        if task is None:
            task = self.task

        # 1. 节点编码
        H = self.node_encoder(S, F)

        # 2. 计算信息载荷
        L, edge_index = compute_load(
            S, F, H,
            edge_gate=self.edge_gate,
            theta=self.theta,
            num_pairs=self.num_pairs,
            eps=self.eps,
            delta=self.delta,
            detour_H=self.detour_H,
            detour_rho=self.detour_rho
        )

        # 3. 生成软掩码
        m = mask_from_L(L, tau=self.tau, threshold=self.threshold)

        # 4. 掩码图Transformer
        Z = self.graph_transformer(H, edge_index, m, eps=self.eps)

        # 5. 任务预测
        y_pred = self.prediction_head(Z, m, edge_index)

        # 6. 损失计算
        loss = None
        loss_dict = {}

        if y is not None:
            _, S_e = build_edge_index_from_S(S)
            a_e = self.edge_gate.last_gate_values

            loss_dict = compute_losses(
                y_pred=y_pred,
                y=y,
                L=L,
                m=m,
                task=task,
                S_e=S_e,
                a_e=a_e,
                lambda_align=self.lambda_align,
                lambda_budget=self.lambda_budget,
                lambda_gate=self.lambda_gate,
                eps=self.eps
            )
            loss = loss_dict['loss']

        # 诊断信息
        diag = {
            'L': L.detach(),
            'm': m.detach(),
            'edge_index': edge_index.detach(),
            'H_node': H.detach(),
            'Z_graph': Z.detach(),
            'y_pred': y_pred.detach() if y is not None else y_pred,
            'loss_dict': loss_dict
        }

        return y_pred, loss, diag

    def inference(
            self,
            S: torch.Tensor,
            F: torch.Tensor,
            k: Optional[int] = None,
            budget_lambda: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """推理模式：使用硬子图选择"""
        self.eval()

        with torch.no_grad():
            H = self.node_encoder(S, F)

            L, edge_index = compute_load(
                S, F, H,
                edge_gate=self.edge_gate,
                theta=self.theta,
                num_pairs=self.num_pairs,
                eps=self.eps,
                delta=self.delta,
                detour_H=self.detour_H,
                detour_rho=self.detour_rho
            )

            edge_index_sub, mask_sub = select_subgraph_from_L(
                L, edge_index, S,
                k=k,
                budget_lambda=budget_lambda
            )

            m_hard = mask_sub.float()
            Z = self.graph_transformer(H, edge_index, m_hard, eps=self.eps)
            y_pred = self.prediction_head(Z, m_hard, edge_index)

        return y_pred, edge_index_sub, mask_sub

    def get_subgraph_importance(
            self,
            S: torch.Tensor,
            F: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取边的重要性分数"""
        self.eval()

        with torch.no_grad():
            H = self.node_encoder(S, F)

            L, edge_index = compute_load(
                S, F, H,
                edge_gate=self.edge_gate,
                theta=self.theta,
                num_pairs=self.num_pairs,
                eps=self.eps,
                delta=self.delta,
                detour_H=self.detour_H,
                detour_rho=self.detour_rho
            )

        return L, edge_index
