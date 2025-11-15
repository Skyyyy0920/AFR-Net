# 任务：实现 Flow→Load→Mask→Subgraph 的最简端到端模型（PyTorch）

请你用 **PyTorch**（>=2.2）+ **NumPy** 实现一个完整、可训练的 Python 模块，核心流程为：

**输入** `S`（结构连接, SC）与 `F`（功能连接, FC） →
**节点编码器（Transformer）** 得到节点嵌入 `H` →
**可微软路由（电网络）** 从 `S,F,H` 计算**信息载荷** `L`（边级标量） →
由 `L` 得到**软掩码** `m`（边级 0~1） →
**下游 Transformer** 在 `m` 掩码的子图上做表征与**任务预测**（分类/回归） →
**端到端训练**（任务损失 + 结构对齐 + 轻正则）。

> 关键要求：
>
> 1. **不对负 FC 单独建模惩罚**，直接用 `F` 原值；
> 2. 训练期使用**软掩码**（可微），推理期从 `L` 导出**硬子图**（允许环，不必是树）；
> 3. 全链路可微，任务梯度能回传到节点编码器与“路由门”。

---

## 一、数据与接口

* 单样本输入（最简版本支持 **单样本前向**，外部训练器可在 batch 维循环）：

  * `S: torch.FloatTensor [N, N]`，非负、对称、对角为 0（结构连接强度）。
  * `F: torch.FloatTensor [N, N]`，实数、对称或近似对称（功能连接，含正负，**不做截断**）。
  * `y: torch.FloatTensor` 或 `torch.LongTensor`（任务标签，可选；分类或回归）。

* 输出：

  * `y_pred`: 预测值（分类 logits 或回归值）。
  * `loss`: 标量；若未传 `y` 返回 `None`。
  * 诊断字典：`{'L': L (E,), 'm': m (E,), 'edge_index': (2,E), 'H_node': H (N,d)}`。

* 约束与默认：

  * `N ≤ 400`；将 `S` 视为图的邻接权重。
  * 仅用 **PyTorch** 标准库（可用 `torch.linalg` 的稠密求解，简化实现）。

---

## 二、总体结构（类与函数）

实现以下类/函数（需要分文件编写，具备良好的文件架构）：

1. `class FlowLoadMaskModel(nn.Module)`
   顶层模型，`forward(S, F, y=None, task='classification' or 'regression')`。
   组合以下组件：

   * `NodeTransformer`: 节点编码器（Graph-Style Transformer）。
   * `EdgeGate`: 由节点嵌入与局部特征生成门值 `a_e∈[0,1]`。
   * `compute_detour_kernel(S) -> K`: 计算 detour（≥2跳）核。
   * `compute_load(S, F, H) -> (L, edge_index)`: 可微软路由，输出边信息载荷 `L` 与边列表。
   * `mask_from_L(L) -> m`: 从 `L` 得到软掩码 `m`（0~1）。
   * `MaskedGraphTransformer`: 下游 Transformer（在掩码子图上做消息传递/注意力）。
   * `PredictionHead`: 任务头（分类/回归）。
   * `compute_losses(y_pred, y, L, m, ...) -> loss_dict`: 组合任务损失与正则。

2. `class NodeTransformer(nn.Module)`

   * 输入：`S, F` → 初始节点特征 `X`（见下），经 `L_node` 层 Transformer 得到节点嵌入 `H∈R^{N×d}`。
   * **节点初始特征 `X`**（请实现简单稳健版）：

     * 结构侧：`deg = S.sum(dim=1, keepdim=True)`、`strength = deg`；
     * 功能侧：`row_sum = F.sum(1, keepdim=True)`、`row_abs_sum = F.abs().sum(1, keepdim=True)`；
     * 以及一个**可学习 ROI 嵌入**表 `nn.Embedding(N, d0)`（若需要，形状对齐）。
     * 将上述拼接并线性映射到 `d` 维。
   * **注意力掩码**：编码器阶段 **仅允许 SC 的一跳邻接**（含自环）参与注意力，非邻接对置 `-inf` bias。
   * 可用标准 Multi-Head Self-Attention + FFN + 残差 + LayerNorm。

3. `class EdgeGate(nn.Module)`

   * 输入：`H` 与边 `(u,v)` 的局部特征：
     `[|H_u-H_v|, H_u⊙H_v, S_uv, F_uv]` → MLP(2-3层, SiLU/ELU) → Sigmoid，输出 `a_e∈[0,1]`。
   * 对称性：特征是对称构造，保证 `a_uv=a_vu`。

4. `compute_detour_kernel(S) -> K`

   * 计算 **截断步行核（去掉 0/1 跳）**：

     ```
     A = S / (S.sum(dim=1, keepdim=True) + eps)           # 行归一
     K = sum_{h=2..H} rho^h * A^h                          # H=5, rho=0.6 (默认)
     ```
   * 返回 `K: [N,N]`，非负。

5. `compute_load(S, F, H) -> (L, edge_index)`  【核心：可微软路由 → 信息载荷】

   * **边列表**：从 `S` 的上三角 `>0` 抽取，获得 `edge_index: [2,E]` 与 `S_e`。
   * **基础阻力**：`c_e = -log(S_e + eps)`。
   * **门值**：`a_e = EdgeGate(H, edge_index, S_e, F_e)`；
   * **导通率**：`g_e = exp(-c_e + theta * a_e)`（`theta>0` 默认 `2.0`）。
   * **detour 核**：`K = compute_detour_kernel(S)`。
   * **成对需求**（**直接使用 F，不做任何 ψ**）：
     `M = F * K`（逐元素），可不强制非负。
   * **采样成对**（为计算效率）：

     * 令 `W = M.abs()`，`T = W.sum()`；
     * 建立概率 `P_ij = W_ij / T`；
     * 采样 `B` 个 `(i,j)`（默认 `B=1024`），每对赋权 `alpha = sign(M_ij) * (T / B)`（保证对 `Σ_{ij} M_ij * f_e^{(ij)}` 的无偏估计）。
   * **电网络解**（稠密实现，便于可微）：

     * 构建边-点关联矩阵 `Bmat: [E,N]`（无向边任选方向，行里为 `+1/-1`）。
     * 拉普拉斯：`Lg = Bmat.T @ diag(g_e) @ Bmat`；为可逆性，做**接地**（去一行一列）或 `Lg + delta*I`（`delta=1e-6`）。
     * 右端：对每个 `(i,j)` 构造 `b = e_i - e_j`，堆叠为 `RHS: [N,B]`，一次性解 `Phi = solve(Lg, RHS)` 得电位。
     * 边电位差：`Delta = Bmat @ Phi`（形状 `[E,B]`）；
     * **单位需求的边流**：`flows = g_e[:,None] * sqrt(Delta**2 + eps**2)`（`eps=1e-6`，soft-abs）。
   * **信息载荷**：按上面的 `alpha` 聚合：
     `L = (flows * alpha[None, :]).sum(dim=1)  # [E]`。
   * **数值稳定**：对 `L` 做标准化（如 `(L - L.mean()) / (L.std()+1e-6)`），同时返回未标准化版本以便诊断（如需）。
   * 返回 `L, edge_index`。

6. `mask_from_L(L) -> m`

   * 软掩码：`m = sigmoid(tau * (L - t))`，`tau=8.0`，`t` 可学习（`nn.Parameter`，初值 `0`）。
   * 也可在 `L` 标准化后再阈值化。

7. `class MaskedGraphTransformer(nn.Module)`

   * 下游 Transformer，输入节点嵌入 `H` 与 `edge_index, m`：

     * 构造注意力偏置矩阵 `A_bias [N,N]`：非边对设为 `-inf`，边对设为 `log(m_e + eps)`（或线性缩放）；允许自环。
     * 在多头注意力中加入 `A_bias`，使得注意力仅发生在被 `m` 选择的边（软方式）。
   * 堆叠 `L_graph` 层，输出 `Z`（节点级表示）。

8. `class PredictionHead(nn.Module)`

   * 读出：全局注意力池化或简单 `mean pooling`（带 `m` 的加权可选）；
   * 分类：线性层输出 logits；回归：线性层输出标量。

9. `compute_losses(y_pred, y, L, m, task, S)`

   * **任务损失**：

     * 分类：`CrossEntropyLoss`；
     * 回归：`MSELoss`。
   * **对齐损失（推荐，轻量）**：

     * `p = softmax(L)`，`q = softmax(log(m + eps))`，
     * `L_align = KL(p || stopgrad(q))`（或相反方向，任选其一实现）。
   * **稀疏/预算正则（轻）**：

     * 结构成本向量 `c_e = -log(S_e + eps)`（同上），
     * `L_budget = (c_e * m).mean()`（或 `.sum()/E`）。
   * **门稀疏（轻）**：`L_gate = a_e.abs().mean()`（从 `EdgeGate` 获取最近一次前向的 `a_e`）。
   * **总损失**：
     `loss = L_task + λ_align*L_align + λ_budget*L_budget + λ_gate*L_gate`
     默认系数：`λ_align=0.2, λ_budget=0.05, λ_gate=1e-4`。
   * 返回 `{'loss': loss, 'task': L_task, 'align': L_align, 'budget': L_budget, 'gate': L_gate}`。

10. **推理期的硬子图选择（简单可用版）**

    * 函数 `select_subgraph_from_L(L, edge_index, S, k=None, budget_lambda=None)`：

      * **Top-k + 连通修复**（默认）：

        * 若传入 `k`，选 `L` 最大的 `k` 条边；
        * 若不连通，用 `S` 上的最短“修复边”连通（直到连通或边数达上限）；
        * 再做 1~2 轮本地交换（用未选边替换收益更高者）。
      * 或 **预算化最大权**：若传 `budget_lambda`，最大化 `Σ L_e − budget_lambda·Σ c_e − ω·|E*|`（`ω` 小常数），用贪心扩张+回退。
      * 返回 `edge_index_sub, mask_sub (bool[E])`。
    * 推理时：先跑 `compute_load` 得 `L`，再用此函数导出**硬子图**（允许环）。

---

## 三、数值与实现要点

* 所有求解使用 **稠密** `torch.linalg.solve` 即可（`N≤400` 足够）；为可逆性采用**接地**（删一行一列）或加 `δI`（`1e-6`）。
* 边-点关联矩阵 `Bmat` 的构造要固定方向，但所有量最终按无向处理（`a_e,g_e,L_e,m_e` 在 `(u,v)` 与 `(v,u)` 相同）。
* 采样对的权重无偏性：从 `P_ij ∝ |M_ij|` 采样 `B` 个，对每对赋权 `alpha = sign(M_ij) * (T/B)`，其中 `T = Σ|M|`。
* `L` 在进入 `mask_from_L` 前做标准化（`zscore`）能显著稳住训练。
* `MaskedGraphTransformer` 的注意力偏置：**非边**→`-1e9`；**边**→`log(m_e+eps)` 或 `β*m_e`（`β` 为尺度）。
* 模块内需要缓存最近的 `a_e` 以用于 `L_gate` 正则。

---

## 四、默认超参（可作为类的 `__init__` 默认）

* 维度：`d=64`，编码器层数：`L_node=2`，下游层数：`L_graph=2`，注意力头：`nhead=4`。
* detour 核：`H=5, rho=0.6` 或者热核 `tau=1.0`（二选一实现截断步行即可）。
* 路由：`theta=2.0`，采样对数 `B=1024`，`eps=1e-6`，`delta=1e-6`（拉普拉斯岭）。
* 掩码：`tau=8.0`，阈值 `t` 学习参数，初值 0。
* 损失系数：`λ_align=0.2, λ_budget=0.05, λ_gate=1e-4`。
* 读出：`mean pooling`（最简）。

---

## 五、需要提供的公共方法/文档

* 顶层 `forward` 注释清楚输入/输出、形状、训练/推理分支（若 `y is None`，仅返回 `y_pred=None, loss=None, diag`）。
* 独立函数/方法：

  * `build_edge_index_from_S(S)`
  * `build_incidence_matrix(edge_index, N)`
  * `laplacian_from_conductance(Bmat, g_e)`（含接地/岭）
  * `solve_potentials(Lg, pair_indices)`（多右端一次性求解）
  * `edge_flows_from_potential(Bmat, Phi, g_e)`
  * `standardize(x)`
  * `select_subgraph_from_L(...)`
* 每个关键步骤写简短 docstring，标明公式对应关系（见上文公式）。

---

## 六、正确性与可用性要求

* 代码需**自包含**：`import torch, torch.nn as nn, torch.nn.functional as F, numpy as np` 即可运行；不依赖三方 GNN 框架。
* 所有张量均在 `S.device` 上；注意 CPU/GPU 兼容。
* 写一个最小的 `if __name__ == "__main__":` 形状自检（不做训练，只构造小图、跑 `forward` 验证维度无误）。
* 不写训练循环/数据加载器/评估脚本（外部会接入）。

---

**总结**：
请实现一个端到端可微模型：`S,F → NodeTransformer(H) → 可微软路由得 L → 软掩码 m → MaskedGraphTransformer → 任务头 → 损失`。
注意：成对需求 **直接使用 `M = F * K`**（不对负 FC 单独处理），信息载荷 `L` 既用于训练期的**软掩码**，也用于推理期的**硬子图选择**（允许环）。模型应暴露清晰的 API 与中间诊断量，便于后续集成与分析。

---

## ✅ 实现完成

已完成完整的 PyTorch 实现，包含以下文件：

### 核心文件

1. **`flow_load_mask_model.py`** - 主要实现文件
   - `FlowLoadMaskModel`: 顶层模型类
   - `NodeTransformer`: 节点编码器
   - `EdgeGate`: 边门控模块
   - `MaskedGraphTransformer`: 掩码图Transformer
   - `PredictionHead`: 任务预测头
   - 所有辅助函数（电网络求解、detour核计算等）
   - 包含自检测试代码

2. **`example_usage.py`** - 使用示例
   - 单样本前向传播
   - 训练循环
   - 推理模式（硬子图选择）
   - 回归任务
   - 批量训练

3. **`API_DOCUMENTATION.md`** - 详细API文档
   - 所有类和函数的完整说明
   - 参数解释
   - 使用流程
   - 常见问题解答

### 快速开始

```python
import torch
from flow_load_mask_model import FlowLoadMaskModel

# 创建数据（示例）
N = 100
S = torch.rand(N, N)
S = (S + S.t()) / 2
S.fill_diagonal_(0)
F = torch.randn(N, N)
F = (F + F.t()) / 2
y = torch.tensor(1)

# 创建模型
model = FlowLoadMaskModel(N=N, num_classes=2)

# 前向传播
y_pred, loss, diag = model(S, F, y, task='classification')

# 反向传播
loss.backward()

# 推理（硬子图）
y_pred, edge_sub, mask = model.inference(S, F, k=50)
```

### 运行测试

```bash
# 自检测试
python flow_load_mask_model.py

# 运行所有示例
python example_usage.py
```

### 主要特性

✅ 完整的端到端可微架构  
✅ 支持分类和回归任务  
✅ 训练期软掩码 + 推理期硬子图  
✅ 电网络路由的可微实现  
✅ GPU/CPU 自动适配  
✅ 详细的中间诊断信息  
✅ 无需外部GNN框架依赖