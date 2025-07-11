
import numpy as np
import pandas as pd
# 示例数据111
import math
from scipy.stats import norm




datas=pd.read_csv('Materials/data/Per1_total.csv')
#print(datas)
data=datas['vr']
#print(data)


# 定义对数似然函数
def log_likelihood(data, mu, sigma,delta_vr):
    n = len(data)
    

    outcome=1
    for i in range(n):
        
        sigma_total=(sigma**2+delta_vr[i]**2)**0.5
        temp=((2*np.pi)**(-0.5))*(sigma_total**-1)*np.exp(-0.5*((data[i]-mu)**2)*(sigma_total**(-2)))
        outcome=outcome*temp
    return outcome


delta_vr=datas['e_vr']
#print(delta_vr)



def index(ori,n):
    #start=ori*0.1
    fendu=ori*0.2/n
    return [ori*0.9+fendu*i for i in range(n)]


def index2(ori,n):
    #start=ori*0.1
    fendu=ori*1.2/n
    return [ori*+0.0+fendu*i for i in range(n)]


#print(index(8.2,20))

result=np.zeros((100,100))
#print(result)
for i in range(result.shape[0]):
    print(i)
    for j in range(result.shape[1]):
        result[i][j]=log_likelihood(data=data,mu=index(-325.98,result.shape[0])[i],sigma=index2(8.57,result.shape[1])[j],delta_vr=delta_vr)


#print(result)
x_labels=index(-325.98,result.shape[0])


#print('y_lable',index2(8.57,result.shape[1]))
y_labels=index2(8.57,result.shape[1])

# 计算最小值和最大值
min_val = np.min(result)
max_val = np.max(result)

# 归一化
result = (result - min_val) / (max_val - min_val)

import matplotlib.pyplot as plt
import seaborn as sns
# # 绘制热力图
# plt.figure(figsize=(8, 6))  # 设置图形大小
# sns.heatmap(result.T, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0, 
#             xticklabels=x_labels, yticklabels=y_labels, cbar_kws={"label": "Intensity"})
# plt.title('Heatmap with Custom Coordinates', fontsize=16)
# plt.xlabel('mu', fontsize=14)
# plt.ylabel('sigma', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(False)

# # 添加等高线
# # 使用 contour 绘制等高线
# contour_levels = [0.01,0.1,0.5,0.99]  # 定义等高线的层级
# contour_plot = plt.contour(result.T, levels=contour_levels, colors='white', linewidths=1.5)

# # 添加等高线标签（可选）
# plt.clabel(contour_plot, inline=True, fontsize=10, fmt="%.2f")


# plt.show()

# 行方向累计（每行累计后的最后一个值）  #mu
row_cumsum_last = np.cumsum(result, axis=1)[:, -1]
col_cumsum_last = np.cumsum(result, axis=0)[-1, :]
#row_cumsum_last=col_cumsum_last

row_cumsum_last=row_cumsum_last/row_cumsum_last.sum()

row_cumsum= np.cumsum(row_cumsum_last, axis=0)
#print(row_cumsum)
#print(row_cumsum_last.sum())
#print("行方向累计结果（一维数组）：")
#print(row_cumsum_last,row_cumsum_last.shape) 

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

left=find_insert_position(row_cumsum,0.84)
right=find_insert_position(row_cumsum,0.16)
middle=find_insert_position(row_cumsum,0.5)

print(left,right)

# 画图
plt.plot(x_labels, row_cumsum_last, marker='o')  # marker='o' 表示每个点用圆圈标记
plt.xlabel('mu')
plt.ylabel('cumsum')
plt.title('mu-cumsum')
plt.grid(True)  # 显示网格
# 添加竖线
plt.axvline(x=x_labels[left], color='r', linestyle='--', linewidth=1, label='16%')
plt.axvline(x=x_labels[right], color='g', linestyle='--', linewidth=1, label='84%')
plt.axvline(x=x_labels[middle], color='b', linestyle='--', linewidth=1, label='50%')
# 显示图例
plt.legend()
print('对于mu方向:',x_labels[left],x_labels[middle],x_labels[right])

print('x=',x_labels[middle],'y=',x_labels[middle]-x_labels[left],'z=',x_labels[right]-x_labels[middle])

plt.show()







# 列方向累计（每列累计后的最后一个值） #sigma
col_cumsum_last = np.cumsum(result, axis=0)[-1, :]
#print("列方向累计结果（一维数组）：")

col_cumsum_last=col_cumsum_last/col_cumsum_last.sum()
##print(col_cumsum_last,col_cumsum_last.shape,col_cumsum_last.sum())

col_cumsum=np.cumsum(col_cumsum_last, axis=0)

left=find_insert_position(col_cumsum,0.84)
right=find_insert_position(col_cumsum,0.16)
middle=find_insert_position(col_cumsum,0.5)

left,right=right,left
# 画图
plt.plot(y_labels, col_cumsum_last, marker='o')  # marker='o' 表示每个点用圆圈标记
plt.xlabel('sigma')
plt.ylabel('cumsum')
plt.title('sigma-cumsum')
plt.grid(True)  # 显示网格


# 添加竖线
plt.axvline(x=y_labels[left], color='r', linestyle='--', linewidth=1, label='16%')
plt.axvline(x=y_labels[right], color='g', linestyle='--', linewidth=1, label='84%')
plt.axvline(x=y_labels[middle], color='b', linestyle='--', linewidth=1, label='50%')


# 显示图例
plt.legend()
print('对于sigma方向:',y_labels[left],y_labels[middle],y_labels[right])

print('x=',y_labels[middle],'y=',y_labels[middle]-y_labels[left],'z=',y_labels[right]-y_labels[middle])


plt.show()