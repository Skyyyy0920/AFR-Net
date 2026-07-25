import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_connectivity_circle

# 1. 准备数据
# 加载你上传的文件
edges_df = pd.read_csv(r'C:\Users\Skyyyy\OneDrive\Desktop\plots_brainconn_stats\Anx_Model_SIGNIFICANT_edges.csv')
roi_df = pd.read_csv(r'C:\Users\Skyyyy\OneDrive\Desktop\plots_brainconn_stats\Glasser_ROI_w_FN.csv')

# 2. 预处理：构建 ROI -> Network 映射
# 注意：根据你的数据，ROI名字是 "L_AreaName" 格式，例如 "L_H"
roi_df['Match_Name'] = roi_df['Hemsphere'] + "_" + roi_df['AreaName']
roi_map = dict(zip(roi_df['Match_Name'], roi_df['NETWORK']))
# 手动修复缺失的 caudate_right
roi_map['caudate_right'] = 'Subcortical'

# 3. 筛选显著边 (P_FDR < 0.05)
sig_edges = edges_df[edges_df['P_FDR'] < 0.05].copy()

# 4. 提取涉及的节点 (Active Nodes)
active_nodes = list(set(sig_edges['ROI_A']).union(set(sig_edges['ROI_B'])))
active_nodes.sort()
n_nodes = len(active_nodes)

# 5. 构建邻接矩阵 (Adjacency Matrix for Plotting)
# 值使用 T-statistic 的绝对值（表示连接强度差异的大小）
con_matrix = np.zeros((n_nodes, n_nodes))
node_to_idx = {node: i for i, node in enumerate(active_nodes)}

for _, row in sig_edges.iterrows():
    i = node_to_idx[row['ROI_A']]
    j = node_to_idx[row['ROI_B']]
    # 使用 T-stat 的绝对值作为连线颜色深度的依据
    con_matrix[i, j] = abs(row['T_Statistic'])
    con_matrix[j, i] = abs(row['T_Statistic'])  # 对称

# 6. 准备节点颜色和标签
# 获取每个节点的网络颜色
# 这里定义一个简单的颜色映射，你可以根据需要修改
network_colors = {
    'Default': 'red',
    'Visual1': 'blue',
    'Visual2': 'cyan',
    'Ventral-Multimodal': 'orange',
    'Subcortical': 'purple',
    'Dorsal-Attention': 'green',
    'Frontoparietal': 'yellow',
    'Auditory': 'pink'
}

# 获取每个节点的 Network 名字
node_networks = [roi_map.get(node, 'Unknown') for node in active_nodes]
# 生成颜色列表
node_colors = [network_colors.get(net, 'gray') for net in node_networks]

# 7. 绘图
# 注意：你需要安装 mne (pip install mne)
fig, ax = plot_connectivity_circle(
    con_matrix,
    node_names=active_nodes,
    node_colors=node_colors,
    title='Hyper-flow Patterns (Anxiety > HC)',
    fontsize_names=8,
    linewidth=2.0,
    colormap='Reds',  # 只用红色系
    vmin=4.0,  # T-stat 最小值 (根据数据调整)
    vmax=10.0,  # T-stat 最大值
    facecolor='white',
    textcolor='black'
)

plt.show()
plt.savefig('Figure5B_Pathology.pdf')
