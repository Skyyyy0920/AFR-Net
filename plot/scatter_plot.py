import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
raw_data = {
    'MLP': [0.6197, '2min'],
    'GCN': [0.4750, '4min'],
    'GAT': [0.5823, '5min'],
    'GIN': [0.4807, '3min'],
    'GraphSAGE': [0.5994, '4min'],
    'Graphormer': [0.6195, '2h'],
    'BrainGNN': [0.5274, '6min'],
    'Triplet': [0.5082, '10min'],
    'AGT': [0.5379, '20min'],
    'BQN': [0.4991, '3min'],
    'Cross-GNN': [0.5625, '5min'],
    'MaskGNN': [0.5809, '5min'],
    'RH-BrainFS': [0.5969, '30min'],
    'NeuroPath': [0.6266, '8min'],
    'Ours': [0.6910, '30min']
}

# 2. 数据解析
methods = []
accs = []
times = []

for name, values in raw_data.items():
    methods.append(name)
    accs.append(values[0])
    t_str = values[1]
    if 'h' in t_str:
        t_val = float(t_str.replace('h', '')) * 60
    elif 'min' in t_str:
        t_val = float(t_str.replace('min', ''))
    else:
        t_val = float(t_str)
    times.append(t_val)

# 3. 开始绘图
plt.figure(figsize=(10, 7), dpi=100)

# 定义颜色
bright_blue = '#007BFF'
bright_red = '#FF0000'

# 绘制 Baselines
plt.scatter(times[:-1], accs[:-1], color=bright_blue, s=80, alpha=0.8, edgecolors='none', label='Baselines')

# 绘制 Ours
plt.scatter(times[-1], accs[-1], color=bright_red, marker='*', s=180, edgecolors='none', label='Ours', zorder=10)

# 4. 添加标签 (精细化位置控制)
for i, txt in enumerate(methods):

    # CASE 1: Graphormer (耗时太长，放在左边)
    if txt == 'Graphormer':
        plt.text(times[i] - 3, accs[i], txt, fontsize=16, ha='right', va='center')

    # CASE 2: Ours (放在右侧，高亮)
    elif txt == 'Ours':
        plt.text(times[i] + 3, accs[i], txt, fontsize=18, fontweight='bold', ha='left', va='center')

    # CASE 3: GAT (放在正上方，避免和 MaskGNN 重叠)
    elif txt == 'GAT' or txt == 'GIN':
        # x坐标不变，y坐标向上偏移 0.01 左右
        plt.text(times[i], accs[i] + 0.002, txt, fontsize=16, ha='center', va='bottom')

    # CASE 4: 其他点 (默认放在右侧)
    else:
        plt.text(times[i] + 2, accs[i], txt, fontsize=16, ha='left', va='center')

# 5. 设置坐标轴
plt.xlabel('Time Cost (min)', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)

# 确保不使用对数坐标
# plt.xscale('log')

plt.legend(loc='lower right', fontsize=18, markerscale=1.5)
plt.tight_layout()
plt.savefig('accuracy_vs_time.pdf', format='pdf', bbox_inches='tight')
plt.show()
