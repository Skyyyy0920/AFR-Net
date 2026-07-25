import pandas as pd
import numpy as np

# ================= 1. 加载文件 =================
# 加载显著边
edges_df = pd.read_csv(r'C:\Users\Skyyyy\OneDrive\Desktop\plots_brainconn_stats\Anx_Model_SIGNIFICANT_edges.csv')

# 加载 Glasser 映射表 (ROI -> Network)
glasser_info = pd.read_csv(r'C:\Users\Skyyyy\OneDrive\Desktop\plots_brainconn_stats\Glasser_ROI_w_FN.csv')

# 加载坐标文件 (ROI -> MNI Coordinates)
# 假设 glasser.node 是空格分隔，且第6列(index 5)是索引ID，前3列是x,y,z
glasser_coords = pd.read_csv(r'C:\Users\Skyyyy\OneDrive\Desktop\plots_brainconn_stats\glasser.node', sep='\s+', header=None)
glasser_coords.columns = ['x', 'y', 'z', 'c3', 'c4', 'ID']

# ================= 2. 构建映射字典 =================
# 构造与 edges 文件一致的 ROI 名字 (例如: "L" + "_" + "V1" = "L_V1")
glasser_info['Full_ROI'] = glasser_info['Hemsphere'] + '_' + glasser_info['AreaName']

# 合并坐标信息 (通过 INDEX/ID)
roi_data = pd.merge(glasser_info, glasser_coords, left_on='INDEX', right_on='ID')

# 创建映射字典
roi_to_network = dict(zip(roi_data['Full_ROI'], roi_data['NETWORK']))
roi_to_coords = dict(zip(roi_data['Full_ROI'], roi_data[['x', 'y', 'z']].values.tolist()))

# 手动补充一些可能缺失的 Subcortical 区域 (如 edges 文件里的 caudate_right)
roi_to_network['caudate_right'] = 'Subcortical'
# 尾状核大致坐标，如果文件里没有
roi_to_coords['caudate_right'] = [13, 15, 10]

# ================= 3. 分析网络分布 (Network Summary) =================
network_pairs = {}
for _, row in edges_df.iterrows():
    net_a = roi_to_network.get(row['ROI_A'], "Unknown")
    net_b = roi_to_network.get(row['ROI_B'], "Unknown")

    # 排序以忽略方向 (A-B 和 B-A 算同一种)
    pair = tuple(sorted([net_a, net_b]))
    network_pairs[pair] = network_pairs.get(pair, 0) + 1

print("\n📊 1. 显著连接的网络分布 (Network Enrichment):")
print(f"{'Network A':<20} <-> {'Network B':<20} | {'Count'}")
print("-" * 55)
for pair, count in sorted(network_pairs.items(), key=lambda x: x[1], reverse=True):
    print(f"{pair[0]:<20} <-> {pair[1]:<20} | {count}")

# ================= 4. 寻找 Hub 并输出坐标 (Hub Validation) =================
# 计算度 (Degree)
node_degree = {}
for roi in list(edges_df['ROI_A']) + list(edges_df['ROI_B']):
    node_degree[roi] = node_degree.get(roi, 0) + 1

# 找出 Top 5 Hubs
top_hubs = sorted(node_degree.items(), key=lambda x: x[1], reverse=True)[:5]

print("\n📍 2. Top Hub Regions (用于 Neurosynth 验证):")
print(f"{'ROI Name':<15} | {'Degree':<6} | {'Network':<15} | {'MNI Coordinates (x, y, z)'}")
print("-" * 75)

for roi, degree in top_hubs:
    net = roi_to_network.get(roi, "Unknown")
    coords = roi_to_coords.get(roi, ["?", "?", "?"])
    coords_str = f"{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}" if isinstance(coords[0], (int, float)) else "N/A"
    print(f"{roi:<15} | {degree:<6} | {net:<15} | {coords_str}")

print("\n💡 下一步行动:")
print("复制上面的 'MNI Coordinates' (例如 -24.0, -18.0, -18.0)，")
print("粘贴到 Neurosynth.org 的 'Locations' 搜索框中，查看关联词条是否包含 'Anxiety', 'Memory', 'Emotion'。")