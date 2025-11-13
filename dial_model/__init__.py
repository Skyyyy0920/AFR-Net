"""
DIAL (Differentiable Information-flow Analysis for Load-based subgraph selection)

Flow→Load→Mask→Subgraph 端到端神经网络模型

主要组件:
- FlowLoadMaskModel: 顶层模型
- NodeTransformer: 节点编码器
- EdgeGate: 边门控
- MaskedGraphTransformer: 掩码图Transformer
- PredictionHead: 任务预测头

工具函数:
- compute_load: 可微软路由计算信息载荷
- mask_from_L: 从载荷生成软掩码
- select_subgraph_from_L: 硬子图选择
- compute_losses: 损失计算
"""

from .model import (
    FlowLoadMaskModel,
    NodeTransformer,
    EdgeGate,
    MaskedGraphTransformer,
    PredictionHead,
    AttentionPooling
)
from .routing import (
    compute_load,
    compute_detour_kernel,
    mask_from_L,
    select_subgraph_from_L
)
from .losses import (
    compute_losses,
    classification_loss,
    regression_loss,
    alignment_loss,
    budget_loss,
    gate_sparsity_loss
)
from .utils import (
    build_edge_index_from_S,
    build_incidence_matrix,
    laplacian_from_conductance,
    solve_potentials,
    edge_flows_from_potential,
    standardize,
    create_attention_mask_from_adjacency,
    select_top_k_edges
)

__version__ = '1.0.0'

__all__ = [
    # 主要模型
    'FlowLoadMaskModel',
    
    # 组件
    'NodeTransformer',
    'EdgeGate',
    'MaskedGraphTransformer',
    'PredictionHead',
    'AttentionPooling',
    
    # 路由函数
    'compute_load',
    'compute_detour_kernel',
    'mask_from_L',
    'select_subgraph_from_L',
    
    # 损失函数
    'compute_losses',
    'classification_loss',
    'regression_loss',
    'alignment_loss',
    'budget_loss',
    'gate_sparsity_loss',
    
    # 工具函数
    'build_edge_index_from_S',
    'build_incidence_matrix',
    'laplacian_from_conductance',
    'solve_potentials',
    'edge_flows_from_potential',
    'standardize',
    'create_attention_mask_from_adjacency',
    'select_top_k_edges',
]

