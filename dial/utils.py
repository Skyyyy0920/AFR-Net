"""
工具函数模块
提供边索引构建、关联矩阵、标准化等基础功能
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def build_edge_index_from_S(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从结构连接矩阵S构建边索引和边权重
    
    参数:
        S: [N, N] 对称邻接矩阵（非负）
        
    返回:
        edge_index: [2, E] 边索引（仅上三角）
        edge_weight: [E] 边权重
    """
    # 只取上三角，避免重复
    triu_mask = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)
    edges = (S > 0) & triu_mask
    
    # 获取边的索引
    row, col = torch.where(edges)
    edge_index = torch.stack([row, col], dim=0)  # [2, E]
    
    # 获取边权重
    edge_weight = S[row, col]  # [E]
    
    return edge_index, edge_weight


def build_incidence_matrix(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    """
    构建边-节点关联矩阵（无向图）
    
    参数:
        edge_index: [2, E] 边索引
        N: 节点数
        
    返回:
        Bmat: [E, N] 关联矩阵，每行对应一条边，+1在源节点，-1在目标节点
    """
    E = edge_index.shape[1]
    device = edge_index.device
    
    Bmat = torch.zeros(E, N, device=device, dtype=torch.float32)
    
    # 对每条边，源节点+1，目标节点-1
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
    从导通率构建图拉普拉斯矩阵（带岭正则）
    
    参数:
        Bmat: [E, N] 关联矩阵
        g_e: [E] 边导通率
        delta: 岭正则化参数
        
    返回:
        Lg: [N, N] 拉普拉斯矩阵（加岭项以保证可逆）
    """
    # Lg = B^T @ diag(g) @ B
    G_diag = torch.diag(g_e)  # [E, E]
    Lg = Bmat.t() @ G_diag @ Bmat  # [N, N]
    
    # 加岭正则化
    N = Lg.shape[0]
    Lg = Lg + delta * torch.eye(N, device=Lg.device, dtype=Lg.dtype)
    
    return Lg


def solve_potentials(
    Lg: torch.Tensor,
    pair_indices: torch.Tensor,
    N: int
) -> torch.Tensor:
    """
    求解多个源-汇对的电位
    
    参数:
        Lg: [N, N] 拉普拉斯矩阵
        pair_indices: [2, B] 源-汇对索引
        N: 节点数
        
    返回:
        Phi: [N, B] 电位矩阵，每列对应一个源-汇对
    """
    B = pair_indices.shape[1]
    device = Lg.device
    
    # 构建右端项矩阵 RHS: [N, B]
    RHS = torch.zeros(N, B, device=device, dtype=Lg.dtype)
    
    batch_idx = torch.arange(B, device=device)
    RHS[pair_indices[0], batch_idx] = 1.0   # 源节点 +1
    RHS[pair_indices[1], batch_idx] = -1.0  # 汇节点 -1
    
    # 求解 Lg @ Phi = RHS
    try:
        Phi = torch.linalg.solve(Lg, RHS)  # [N, B]
    except RuntimeError as e:
        print(f"警告: 拉普拉斯求解失败，使用最小二乘: {e}")
        Phi = torch.linalg.lstsq(Lg, RHS).solution
    
    return Phi


def edge_flows_from_potential(
    Bmat: torch.Tensor,
    Phi: torch.Tensor,
    g_e: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    从电位计算边上的流（信息载荷的基础）
    
    参数:
        Bmat: [E, N] 关联矩阵
        Phi: [N, B] 电位
        g_e: [E] 边导通率
        eps: 数值稳定性参数
        
    返回:
        flows: [E, B] 每条边在每个源-汇对下的流量
    """
    # 计算电位差 Delta = Bmat @ Phi: [E, B]
    Delta = Bmat @ Phi
    
    # 流 = 导通率 × 软绝对值(电位差)
    # 使用 sqrt(Delta^2 + eps^2) 作为可微的绝对值
    flows = g_e.unsqueeze(1) * torch.sqrt(Delta ** 2 + eps ** 2)  # [E, B]
    
    return flows


def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    标准化（Z-score）
    
    参数:
        x: 输入张量
        eps: 数值稳定性参数
        
    返回:
        标准化后的张量
    """
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def create_attention_mask_from_adjacency(
    S: torch.Tensor,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    从邻接矩阵创建注意力掩码
    
    参数:
        S: [N, N] 邻接矩阵
        add_self_loops: 是否添加自环
        
    返回:
        mask: [N, N] 注意力掩码（非邻接对为 -inf）
    """
    N = S.shape[0]
    device = S.device
    
    # 创建邻接掩码（S > 0 的位置）
    adj_mask = (S > 0).float()
    
    if add_self_loops:
        # 添加自环
        adj_mask = adj_mask + torch.eye(N, device=device, dtype=adj_mask.dtype)
    
    # 将0位置设为-inf，1位置设为0
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
    选择Top-K边（可选连通性修复）
    
    参数:
        L: [E] 边的载荷/重要性分数
        k: 选择的边数
        ensure_connected: 是否确保连通性
        S: [N, N] 结构连接矩阵（用于连通性修复）
        edge_index: [2, E] 边索引
        
    返回:
        mask: [E] bool张量，True表示选中的边
    """
    E = L.shape[0]
    k = min(k, E)
    
    # 选择Top-K
    _, top_indices = torch.topk(L, k)
    mask = torch.zeros(E, dtype=torch.bool, device=L.device)
    mask[top_indices] = True
    
    # TODO: 这里可以添加连通性修复逻辑
    # 简化版本暂时不做修复
    
    return mask

