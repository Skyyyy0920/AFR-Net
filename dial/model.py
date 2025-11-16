import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder

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
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            num_spatial: int = 510,
            max_degree: int = 511,  # +1 = 512
            max_path_len: int = 5,
            edge_feat_dim: int = 1,
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.nhead = nhead

        self.fc_feature_proj = nn.Sequential(
            nn.Linear(N, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=d_model, direction='both')
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=nhead)
        self.path_encoder = PathEncoder(max_len=max_path_len, feat_dim=edge_feat_dim, num_heads=nhead)

        # Virtual node (graph token)
        self.graph_token = nn.Embedding(1, d_model)
        self.graph_token_virtual_distance = nn.Embedding(1, nhead)

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
        batch_size, num_nodes, _ = node_feat.shape

        # Project node features
        x = self.fc_feature_proj(node_feat)  # [batch, N, d_model]

        # Add degree encoding
        degree_emb = self.degree_encoder(torch.stack((in_degree, out_degree), dim=0))
        H = x + degree_emb

        # Add virtual node
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # [batch, 1, d_model]
        H = torch.cat([graph_token_feat, H], dim=1)  # [batch, N+1, d_model]

        # Prepare attention bias
        attn_bias = torch.zeros(
            batch_size,
            num_nodes + 1,
            num_nodes + 1,
            self.nhead,
            device=dist.device,
        )  # [batch_size, N+1, N+1, attention_head]

        dist_long = dist.long()
        path_encoding = self.path_encoder(dist_long, path_data)
        spatial_encoding = self.spatial_encoder(dist_long)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        # Virtual node spatial encoding
        t = self.graph_token_virtual_distance.weight.reshape(1, 1, self.nhead)
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        for layer in self.layers:
            H = layer(H, attn_mask=attn_mask, attn_bias=attn_bias)

        # # 返回节点 embeddings（去掉 virtual node）
        # node_embeddings = x[:, 1:, :]  # [batch, N, d_model]
        # return self.norm(node_embeddings)
        # # 返回图 embedding（只要 virtual node）
        # graph_embedding = x[:, 0, :]  # [batch, d_model]
        # return self.norm(graph_embedding)

        return self.norm(H[:, 1:, :])  # Remove virtual node


class EdgeGate(nn.Module):
    """边门控：为每条边生成门值 a_e ∈ [0, 1]"""

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        edge_feature_dim = 2 * d_model + 2

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
            H_diff,  # [E, d]
            H_prod,  # [E, d]
            S_e.unsqueeze(1),  # [E, 1]
            F_e.unsqueeze(1)  # [E, 1]
        ], dim=1)

        a_e = self.mlp(edge_features).squeeze(1)  # [E, 1]
        self.last_gate_values = a_e
        return a_e


# ============================================================================
# Transformer with mask to get final representation
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
        attn_bias = self._build_attention_bias(H.shape[0], edge_index, m, eps, H.device)

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
# Prediction Head
# ============================================================================


class PredictionHead(nn.Module):
    def __init__(
            self,
            d_model: int = 64,
            num_classes: int = 2,
            task: str = 'classification',
            hidden_dim: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.task = task

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
            raise ValueError(f"Unsupported task: {task}")

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        graph_repr = Z.mean(dim=0)

        y_pred = self.mlp(graph_repr)

        if self.task == 'regression':
            y_pred = y_pred.squeeze(-1)

        return y_pred


# ============================================================================
# 顶层模型
# ============================================================================

class DIALModel(nn.Module):
    """Flow→Load→Mask→Subgraph 端到端模型"""

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
            theta: float = 2.0,
            num_pairs: int = 1024,
            detour_H: int = 5,
            detour_rho: float = 0.6,
            tau: float = 8.0,
            lambda_align: float = 0.2,
            lambda_budget: float = 0.05,
            lambda_gate: float = 1e-4,
            delta: float = 1e-6
    ):
        super().__init__()

        self.N = N
        self.d_model = d_model
        self.task = task
        self.theta = theta
        self.num_pairs = num_pairs
        self.detour_H = detour_H
        self.detour_rho = detour_rho
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_budget = lambda_budget
        self.lambda_gate = lambda_gate
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
            hidden_dim=dim_feedforward // 2
        )

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
        batch_size = node_feat.shape[0]
        targets = y.squeeze(-1) if y is not None and y.dim() > 1 else y
        H = self.node_encoder(node_feat, in_degree, out_degree, path_data, dist, attn_mask)

        y_pred_list: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        diag_list: list[Dict] = []

        L, edge_index = compute_load(
            S, F, H,
            edge_gate=self.edge_gate,
            theta=self.theta,
            num_pairs=self.num_pairs,
            delta=self.delta,
            detour_H=self.detour_H,
            detour_rho=self.detour_rho
        )

        m = mask_from_L(L, tau=self.tau, threshold=self.threshold)

        Z = self.graph_transformer(H, edge_index, m)

        y_pred = self.prediction_head(Z)

        for idx in range(batch_size):
            H_i = H[idx]
            S_i = S[idx]
            F_i = F[idx]

            L, edge_index = compute_load(
                S_i, F_i, H_i,
                edge_gate=self.edge_gate,
                theta=self.theta,
                num_pairs=self.num_pairs,
                delta=self.delta,
                detour_H=self.detour_H,
                detour_rho=self.detour_rho
            )

            m = mask_from_L(L, tau=self.tau, threshold=self.threshold)
            Z = self.graph_transformer(H_i, edge_index, m)
            y_pred = self.prediction_head(Z)

            loss = None
            loss_dict: Dict[str, float] = {}
            if targets is not None:
                target = targets[idx] if targets.dim() > 0 else targets
                _, S_e = build_edge_index_from_S(S_i)
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
                )
                loss = loss_dict['loss']
                losses.append(loss)

            diag_list.append({
                'L': L.detach(),
                'm': m.detach(),
                'edge_index': edge_index.detach(),
                'H_node': H.detach(),
                'Z_graph': Z.detach(),
                'y_pred': y_pred.detach(),
                'loss_dict': loss_dict
            })
            y_pred_list.append(y_pred.unsqueeze(0))

        y_pred_batch = torch.cat(y_pred_list, dim=0)
        avg_loss = torch.stack(losses).mean() if losses else None
        return y_pred_batch, avg_loss, {'batch_diag': diag_list}

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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """推理模式：使用硬子图选择"""
        self.eval()

        with torch.no_grad():
            H_batch = self.node_encoder(node_feat, in_degree, out_degree, path_data, dist, attn_mask)
            preds = []
            edge_indices = []
            masks = []

            for idx in range(node_feat.shape[0]):
                H = H_batch[idx]
                S_i = S[idx]
                F_i = F[idx]

                L, edge_index = compute_load(
                    S_i, F_i, H,
                    edge_gate=self.edge_gate,
                    theta=self.theta,
                    num_pairs=self.num_pairs,
                    delta=self.delta,
                    detour_H=self.detour_H,
                    detour_rho=self.detour_rho
                )

                edge_index_sub, mask_sub = select_subgraph_from_L(
                    L, edge_index, S_i,
                    k=k,
                    budget_lambda=budget_lambda
                )

                m_hard = mask_sub.float()
                Z = self.graph_transformer(H, edge_index, m_hard)
                y_pred = self.prediction_head(Z)

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
            H_batch = self.node_encoder(node_feat, in_degree, out_degree, path_data, dist, attn_mask)
            loads = []
            edges = []

            for idx in range(node_feat.shape[0]):
                H = H_batch[idx]
                L, edge_index = compute_load(
                    S[idx], F[idx], H,
                    edge_gate=self.edge_gate,
                    theta=self.theta,
                    num_pairs=self.num_pairs,
                    delta=self.delta,
                    detour_H=self.detour_H,
                    detour_rho=self.detour_rho
                )
                loads.append(L)
                edges.append(edge_index)

            if len(loads) == 1:
                return loads[0], edges[0]
            return loads, edges
