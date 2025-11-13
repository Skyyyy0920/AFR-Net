"""
数据加载和预处理模块

包含：
- BrainConnectivityDataset: 数据集类
- load_data: 数据加载
- preprocess_labels: 标签预处理
- balance_dataset: 数据平衡
- split_dataset: 数据集划分
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


class BrainConnectivityDataset(Dataset):
    """脑连接数据集"""
    
    def __init__(self, data_list: List[Dict], device='cpu'):
        """
        参数:
            data_list: 数据列表，每个元素包含 'SC', 'FC', 'label', 'name'
            device: 设备
        """
        self.data_list = data_list
        self.device = device
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 转换为Tensor
        S = torch.from_numpy(item['SC']).float().to(self.device)
        F = torch.from_numpy(item['FC']).float().to(self.device)
        y = torch.tensor(item['label'], dtype=torch.long).to(self.device)
        
        return S, F, y, item['name']


def load_data(data_path: str) -> Dict:
    """
    加载数据
    
    参数:
        data_path: 数据文件路径（pickle格式）
        
    返回:
        data_dict: 原始数据字典
    """
    print(f"[数据加载] 从 {data_path} 加载数据...")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"[数据加载] 加载完成，共 {len(data_dict)} 个样本")
    return data_dict


def preprocess_labels(data_dict: Dict, task: str = 'OCD') -> Dict:
    """
    预处理标签
    
    参数:
        data_dict: 原始数据字典
        task: 任务类型 'OCD' 或 'ADHD_ODD_Cond'
        
    返回:
        processed_dict: 处理后的数据字典
    """
    # 标签索引映射
    label_names = ['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD', 'Eat', 'ADHD', 'ODD', 'Cond', 'PTSD']
    label_idx = {name: i for i, name in enumerate(label_names)}
    
    processed_dict = {}
    
    if task == 'OCD':
        # OCD任务：直接使用OCD标签
        target_idx = label_idx['OCD']
        print(f"[标签处理] OCD任务 - 使用标签索引 {target_idx}")
        
        for key, value in data_dict.items():
            label = value['label'][target_idx]
            processed_dict[key] = {
                'SC': value['SC'],
                'FC': value['FC'],
                'label': label,
                'name': value['name'],
                'original_labels': value['label']
            }
            
    elif task == 'ADHD_ODD_Cond':
        # ADHD/ODD/Cond合并任务：只要有一个是1就是1
        adhd_idx = label_idx['ADHD']
        odd_idx = label_idx['ODD']
        cond_idx = label_idx['Cond']
        print(f"[标签处理] ADHD/ODD/Cond合并任务 - 使用标签索引 {adhd_idx}, {odd_idx}, {cond_idx}")
        
        for key, value in data_dict.items():
            labels = value['label']
            # 只要有一个是1就是1
            merged_label = int(labels[adhd_idx] == 1 or labels[odd_idx] == 1 or labels[cond_idx] == 1)
            processed_dict[key] = {
                'SC': value['SC'],
                'FC': value['FC'],
                'label': merged_label,
                'name': value['name'],
                'original_labels': value['label']
            }
    else:
        raise ValueError(f"不支持的任务类型: {task}")
    
    return processed_dict


def balance_dataset(data_dict: Dict, ratio: float = 1.0, random_state: int = 42) -> Dict:
    """
    平衡数据集（降采样多数类）
    
    参数:
        data_dict: 数据字典
        ratio: 正负样本比例（正样本:负样本）
        random_state: 随机种子
        
    返回:
        balanced_dict: 平衡后的数据字典
    """
    # 分离正负样本
    positive_samples = {k: v for k, v in data_dict.items() if v['label'] == 1}
    negative_samples = {k: v for k, v in data_dict.items() if v['label'] == 0}
    
    n_pos = len(positive_samples)
    n_neg = len(negative_samples)
    
    print(f"[数据平衡] 原始分布 - 正样本: {n_pos}, 负样本: {n_neg}")
    
    # 计算需要的样本数
    if n_pos > n_neg:
        # 正样本多，降采样正样本
        target_pos = int(n_neg * ratio)
        np.random.seed(random_state)
        selected_pos_keys = np.random.choice(list(positive_samples.keys()), target_pos, replace=False)
        balanced_dict = {k: positive_samples[k] for k in selected_pos_keys}
        balanced_dict.update(negative_samples)
    else:
        # 负样本多，降采样负样本
        target_neg = int(n_pos / ratio)
        np.random.seed(random_state)
        selected_neg_keys = np.random.choice(list(negative_samples.keys()), target_neg, replace=False)
        balanced_dict = {k: negative_samples[k] for k in selected_neg_keys}
        balanced_dict.update(positive_samples)
    
    # 统计平衡后的分布
    n_pos_new = sum(1 for v in balanced_dict.values() if v['label'] == 1)
    n_neg_new = sum(1 for v in balanced_dict.values() if v['label'] == 0)
    print(f"[数据平衡] 平衡后分布 - 正样本: {n_pos_new}, 负样本: {n_neg_new}")
    
    return balanced_dict


def split_dataset(data_dict: Dict, test_size: float = 0.3, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    划分训练集和测试集
    
    参数:
        data_dict: 数据字典
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        train_data: 训练集列表
        test_data: 测试集列表
    """
    # 转换为列表
    data_list = list(data_dict.values())
    labels = [item['label'] for item in data_list]
    
    # 分层划分
    train_data, test_data = train_test_split(
        data_list, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # 保持类别比例
    )
    
    # 统计
    train_pos = sum(1 for item in train_data if item['label'] == 1)
    train_neg = len(train_data) - train_pos
    test_pos = sum(1 for item in test_data if item['label'] == 1)
    test_neg = len(test_data) - test_pos
    
    print(f"[数据划分] 训练集: {len(train_data)} (正:{train_pos}, 负:{train_neg})")
    print(f"[数据划分] 测试集: {len(test_data)} (正:{test_pos}, 负:{test_neg})")
    
    return train_data, test_data

