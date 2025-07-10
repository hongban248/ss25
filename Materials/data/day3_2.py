import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

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


path='Materials/data/data1_dg.csv'
datas=pd.read_csv(path)
sigma=datas['sigmas']
mu=datas['mu']
sigma=mu

#print(mu)


# 设置区间数量（你可以根据需要调整）
bins = 40

# 使用 numpy.histogram 统计每个区间的数量
counts, bin_edges = np.histogram(sigma, bins=bins)

# 计算每个区间的中值（中点）
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 打印结果
# for i in range(len(counts)):
#     print(f"区间 {i+1}: 中值 = {bin_centers[i]:.2f}, 数量 = {counts[i]}")

count_cumsum=np.cumsum(counts, axis=0)
count_cumsum=count_cumsum/count_cumsum.max()
#print(count_cumsum)

left=find_insert_position(count_cumsum,0.16)
middle=find_insert_position(count_cumsum,0.5)
right=find_insert_position(count_cumsum,0.84)

#plt.plot(bin_centers,counts, marker='o')
plt.hist(sigma, bins=bins, edgecolor='black', alpha=0.7)
plt.xlabel('mu')
plt.ylabel('cumsum')
plt.title('mu-cumsum')
plt.grid(True)  # 显示网格


plt.axvline(x=bin_centers[left], color='b', linestyle='--', linewidth=1, label='16%')
plt.axvline(x=bin_centers[middle], color='g', linestyle='--', linewidth=1, label='50%')
plt.axvline(x=bin_centers[right], color='r', linestyle='--', linewidth=1, label='84%')
# 显示图例
plt.legend()

print('for mu:',bin_centers[left],bin_centers[middle],bin_centers[right])
print('x=',bin_centers[middle],'y=',bin_centers[middle]-bin_centers[left],'z=',bin_centers[right]-bin_centers[middle])

plt.show()



sigma=datas['sigmas']
# 使用 numpy.histogram 统计每个区间的数量
counts, bin_edges = np.histogram(sigma, bins=bins)

# 计算每个区间的中值（中点）
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 打印结果
# for i in range(len(counts)):
#     print(f"区间 {i+1}: 中值 = {bin_centers[i]:.2f}, 数量 = {counts[i]}")

count_cumsum=np.cumsum(counts, axis=0)
count_cumsum=count_cumsum/count_cumsum.max()
#print(count_cumsum)

left=find_insert_position(count_cumsum,0.16)
middle=find_insert_position(count_cumsum,0.5)
right=find_insert_position(count_cumsum,0.84)

#plt.plot(bin_centers,counts, marker='o')
plt.hist(sigma, bins=bins, edgecolor='black', alpha=0.7)
plt.xlabel('sigma')
plt.ylabel('cumsum')
plt.title('sigma-cumsum')
plt.grid(True)  # 显示网格


plt.axvline(x=bin_centers[left], color='b', linestyle='--', linewidth=1, label='16%')
plt.axvline(x=bin_centers[middle], color='g', linestyle='--', linewidth=1, label='50%')
plt.axvline(x=bin_centers[right], color='r', linestyle='--', linewidth=1, label='84%')
# 显示图例
plt.legend()

print('for sigma:',bin_centers[left],bin_centers[middle],bin_centers[right])
print('x=',bin_centers[middle],'y=',bin_centers[middle]-bin_centers[left],'z=',bin_centers[right]-bin_centers[middle])

plt.show()