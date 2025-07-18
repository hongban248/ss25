import csv
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


rh_path='rh_chains/rh_chains/PerI_rh_chain.csv'
rh_data=pd.read_csv(rh_path)['col6']

 # 读取 rh 链数据的 col6 列
#print(rh_data[1],rh_data.shape)

data=[]
data_log=[]


 # 采样 3000 次，模拟 MCMC 链
for _ in range(3000):
    # 随机选取 rh 链中的一个样本
    rh = rh_data[random.randint(0, len(rh_data))]
    #print(rh)

    # 从正态分布采样 mu
    mu = np.random.normal(24.49, 0.18)
    #print(mu)

    # 计算距离 dc
    dc = 10 ** ((mu + 5) * 0.2)

    # 计算 rh_pc
    rh_pc = dc * (rh * np.pi / (60 * 180))
    #print(dc, rh_pc)

    # 计算 M
    M = rh_pc * 580 * (4.31 ** 2)
    data.append(M)
    data_log.append(np.log10(M))

#print(data)
# 取对数后的 M 作为分析数据
data = data_log

def find_insert_position(arr, target):
    """
    使用二分查找算法查找目标数字在升序数组中的插入位置。
    如果目标数字已经在数组中，返回其索引；如果目标数字不在数组中，
    返回目标数字应该插入的位置，使得插入后数组仍然保持升序。

    :param arr: 升序数组
    :param target: 要查找的目标数字
    :return: 目标数字在数组中的插入位置
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2  # 计算中间索引

        if arr[mid] == target:
            return mid  # 找到目标数字，返回其索引
        elif arr[mid] < target:
            left = mid + 1  # 目标在右侧子数组
        else:
            right = mid - 1  # 目标在左侧子数组

    # 如果未找到目标数字，left 的位置即为插入位置
    return left





# 设置区间数量（你可以根据需要调整）
bins = 100


# 使用 numpy.histogram 统计每个区间的数量
counts, bin_edges = np.histogram(data, bins=bins)

# 计算每个区间的中值（中点）
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 打印每个区间的中值和数量
for i in range(len(counts)):
    print(f"区间 {i+1}: 中值 = {bin_centers[i]:.2f}, 数量 = {counts[i]}")

# 计算累计分布（归一化）
count_cumsum = np.cumsum(counts, axis=0)
count_cumsum = count_cumsum / count_cumsum.max()
#print(count_cumsum)

# 查找累计分布中 16%、50%、84% 的位置
left = find_insert_position(count_cumsum, 0.16)
middle = find_insert_position(count_cumsum, 0.5)
right = find_insert_position(count_cumsum, 0.84)

# 绘制直方图
plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
plt.xlabel('M(or log10 M)')
plt.ylabel('cumsum')
plt.title('log10 M-cumsum')
plt.grid(True)  # 显示网格

# 标出 16%、50%、84% 的分位线
plt.axvline(x=bin_centers[left], color='b', linestyle='--', linewidth=1, label='16%')
plt.axvline(x=bin_centers[middle], color='g', linestyle='--', linewidth=1, label='50%')
plt.axvline(x=bin_centers[right], color='r', linestyle='--', linewidth=1, label='84%')
# 显示图例
plt.legend()

# 打印分位点信息
print('for mu:', bin_centers[left], bin_centers[middle], bin_centers[right])
print('x=', bin_centers[middle], 'y=', bin_centers[middle] - bin_centers[left], 'z=', bin_centers[right] - bin_centers[middle])

# 显示图形
plt.show()
