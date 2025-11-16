"""
路由和载荷计算模块
实现可微软路由（电网络模型）计算信息载荷
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
    计算截断步行核（去掉0/1跳）
    
    K = sum_{h=2..H} rho^h * A^h
    其中 A 是行归一化的转移矩阵
    
    参数:
        S: [N, N] 结构连接矩阵
        H: 最大跳数
        rho: 衰减因子
        eps: 数值稳定性参数
        
    返回:
        K: [N, N] detour核矩阵
    """
    A = S
    
    # 计算 A^h 的累加
    K = torch.zeros_like(S)
    A_power = A @ A  # 从 A^2 开始
    
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
    detour_rho: float = 0.6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可微软路由：计算边信息载荷
    
    流程:
    1. 从S构建边列表
    2. 计算基础阻力 c_e = -log(S_e + eps)
    3. 通过EdgeGate得到门值 a_e
    4. 计算导通率 g_e = exp(-c_e + theta * a_e)
    5. 计算detour核 K
    6. 计算成对需求 M = F * K
    7. 采样成对并求解电网络
    8. 聚合得到载荷 L
    
    参数:
        S: [N, N] 结构连接
        F: [N, N] 功能连接
        H: [N, d] 节点嵌入
        edge_gate: EdgeGate模块
        theta: 门值影响系数
        num_pairs: 采样的源-汇对数量
        eps: 数值稳定性参数
        delta: 拉普拉斯岭正则化
        detour_H: detour核最大跳数
        detour_rho: detour核衰减因子
        
    返回:
        L: [E] 边信息载荷（标准化后）
        edge_index: [2, E] 边索引
    """
    N = S.shape[0]
    device = S.device
    
    # 1. 构建边列表（仅上三角）
    from .utils import build_edge_index_from_S
    edge_index, S_e = build_edge_index_from_S(S)  # [2, E], [E]
    E = edge_index.shape[1]
    
    # 获取边的功能连接
    F_e = F[edge_index[0], edge_index[1]]  # [E]
    
    # 2. 基础阻力
    c_e = -torch.log(S_e + eps)  # [E]
    
    # 3. 门值
    a_e = edge_gate(H, edge_index, S_e, F_e)  # [E]
    
    # 4. 导通率
    g_e = torch.exp(-c_e + theta * a_e)  # [E]
    
    # 5. 计算detour核
    K = compute_detour_kernel(S, H=detour_H, rho=detour_rho)  # [N, N]
    
    # 6. 成对需求
    M = F * K  # [N, N]
    
    # 7. 采样成对
    # 使用 |M| 作为采样权重
    W = M.abs()  # [N, N]
    T = W.sum()  # 标量
    
    # 避免全零的情况
    if T < eps:
        # 如果需求矩阵为空，返回零载荷
        L = torch.zeros(E, device=device)
        return standardize(L), edge_index
    
    # 构建采样概率
    P = W / T  # [N, N]
    
    # 采样（放回）
    num_pairs = min(num_pairs, N * (N - 1) // 2)  # 不超过总对数
    
    # 将P展平并采样
    P_flat = P.reshape(-1)  # [N*N] - 使用reshape而不是view，避免非连续张量问题
    sampled_flat_idx = torch.multinomial(P_flat, num_pairs, replacement=True)  # [num_pairs]
    
    # 转换为2D索引
    sampled_i = sampled_flat_idx // N
    sampled_j = sampled_flat_idx % N
    pair_indices = torch.stack([sampled_i, sampled_j], dim=0)  # [2, num_pairs]
    
    # 每对的权重 alpha = sign(M_ij) * (T / num_pairs)
    M_sampled = M[sampled_i, sampled_j]  # [num_pairs]
    alpha = torch.sign(M_sampled) * (T / num_pairs)  # [num_pairs]
    
    # 8. 电网络求解
    # 构建关联矩阵
    Bmat = build_incidence_matrix(edge_index, N)  # [E, N]
    
    # 构建拉普拉斯
    Lg = laplacian_from_conductance(Bmat, g_e, delta=delta)  # [N, N]
    
    # 求解电位
    Phi = solve_potentials(Lg, pair_indices, N)  # [N, num_pairs]
    
    # 计算边流
    flows = edge_flows_from_potential(Bmat, Phi, g_e, eps=eps)  # [E, num_pairs]
    
    # 9. 聚合载荷
    L_raw = (flows * alpha.unsqueeze(0)).sum(dim=1)  # [E]
    
    # 10. 标准化
    L = standardize(L_raw, eps=eps)  # [E]
    
    return L, edge_index


def mask_from_L(L: torch.Tensor, tau: float = 8.0, threshold: nn.Parameter = None) -> torch.Tensor:
    """
    从载荷L生成软掩码m
    
    m = sigmoid(tau * (L - t))
    
    参数:
        L: [E] 边载荷
        tau: 温度参数
        threshold: 可学习阈值（如果为None，使用0）
        
    返回:
        m: [E] 软掩码，范围 [0, 1]
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
    从载荷L选择硬子图（推理期使用）
    
    支持两种策略:
    1. Top-k: 选择载荷最大的k条边
    2. 预算化: 基于预算约束的贪心选择
    
    参数:
        L: [E] 边载荷
        edge_index: [2, E] 边索引
        S: [N, N] 结构连接（用于连通性修复）
        k: Top-k边数（如果指定）
        budget_lambda: 预算惩罚系数（如果指定）
        
    返回:
        edge_index_sub: [2, E_sub] 选中的边索引
        mask_sub: [E] bool掩码
    """
    E = L.shape[0]
    
    if k is not None:
        # Top-k策略
        k = min(k, E)
        _, top_indices = torch.topk(L, k)
        mask_sub = torch.zeros(E, dtype=torch.bool, device=L.device)
        mask_sub[top_indices] = True
    else:
        # 简单阈值策略（选择正载荷）
        mask_sub = L > 0
    
    # 提取子图边索引
    edge_index_sub = edge_index[:, mask_sub]
    
    return edge_index_sub, mask_sub

