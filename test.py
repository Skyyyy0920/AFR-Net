import scipy.io as sio
import numpy as np

# 读取MAT文件
mat_data = sio.loadmat(r'W:\Brain Analysis\DIAL\data\Vritual\FC_MFM.mat')

# 查看文件中的所有变量名
print("=" * 60)
print("MAT文件中的变量:")
print("=" * 60)
for key in mat_data.keys():
    if not key.startswith('__'):  # 忽略元数据
        print(f"\n变量名: {key}")
        print(f"类型: {type(mat_data[key])}")
        print(f"形状: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
        print(f"数据类型: {mat_data[key].dtype if hasattr(mat_data[key], 'dtype') else 'N/A'}")

# 详细查看每个变量的内容
print("\n" + "=" * 60)
print("详细数据内容:")
print("=" * 60)

for key in mat_data.keys():
    if not key.startswith('__'):
        print(f"\n{'=' * 40}")
        print(f"变量: {key}")
        print(f"{'=' * 40}")
        data = mat_data[key]

        # 如果是数组，显示一些统计信息
        if isinstance(data, np.ndarray):
            print(f"维度: {data.ndim}")
            print(f"形状: {data.shape}")
            print(f"总元素数: {data.size}")

            # 如果是数值型数据，显示统计信息
            if np.issubdtype(data.dtype, np.number):
                print(f"最小值: {np.min(data)}")
                print(f"最大值: {np.max(data)}")
                print(f"平均值: {np.mean(data)}")
                print(f"标准差: {np.std(data)}")

            # 显示数据预览（前几个元素）
            print(f"\n数据预览:")
            if data.size <= 20:
                print(data)
            else:
                print(f"数据过大，显示部分内容:")
                if data.ndim == 1:
                    print(data[:10])
                elif data.ndim == 2:
                    print(data[:5, :5])
                else:
                    print(f"多维数据，形状为: {data.shape}")
        else:
            print(data)

# 可选：将某个变量保存为单独的变量使用
# 例如：如果文件中有名为 'data' 的变量
# my_data = mat_data['data']
# print("\n提取的数据:")
# print(my_data)