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
print(delta_vr)



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
    #print(i)
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
# 绘制热力图
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.heatmap(result.T, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0, 
            xticklabels=x_labels, yticklabels=y_labels, cbar_kws={"label": "Intensity"})
plt.title('Heatmap with Custom Coordinates', fontsize=16)
plt.xlabel('mu', fontsize=14)
plt.ylabel('sigma', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)

# 添加等高线
# 使用 contour 绘制等高线
contour_levels = [0.01,0.1,0.5,0.99]  # 定义等高线的层级
contour_plot = plt.contour(result.T, levels=contour_levels, colors='white', linewidths=1.5)

# 添加等高线标签（可选）
plt.clabel(contour_plot, inline=True, fontsize=10, fmt="%.2f")



import random
x=random.randint(0, result.shape[1]-1)  #mu
y=random.randint(0, result.shape[0]-1)   #sigma
plt.scatter(x, y, color='red', s=50, marker='o',label='start')
# x=random.randint(0, 10)  #mu
# y=random.randint(0, 10)   #sigma

a=random.random()   
#print(x,y,a)

mcmc_list=[]
mcmc_list.append((x,y))

# 设置收敛阈值
need_convergence_threshold = 12  # 对数似然值变化的阈值
max_iterations = 10000  # 最大迭代次数

flag=0

for i in range(max_iterations):
    l0=log_likelihood(data=data,mu=x_labels[x],sigma=y_labels[y],delta_vr=delta_vr)
    #print(l0)

    # 随机选择附近的一个点
    # new_x = x + random.randint(-1, 1)  # 在 x 的邻域内随机选择
    # new_y = y + random.randint(-1, 1)  # 在 y 的邻域内随机选择

    # 随机选择附近的一个点，随机步长
    #step_size = abs(int(np.random.normal(0, y_labels[y])) ) # 随机步长范围为
    step_size=12 
    new_x = x + int(np.random.normal(0, 12) )# 在 x 的邻域内随机选择 #6
    new_y = y + int(np.random.normal(0, 13.3) )  # 在 y 的邻域内随机选择 #2

    # 确保新点在合法范围内
    new_x = max(0, min(new_x, result.shape[1] - 1))
    new_y = max(0, min(new_y, result.shape[0] - 1))

    l1 = log_likelihood(data=data, mu=x_labels[new_x], sigma=y_labels[new_y],delta_vr=delta_vr)

    if l1>l0:
        x=new_x
        y=new_y
        mcmc_list.append((x,y))
        flag=flag+1
    else:
        a=random.random()
        if a < l1/l0:
            x=new_x
            y=new_y
            mcmc_list.append((x,y))
            flag=flag+1
        else:
            mcmc_list.append((x,y))
    plt.scatter(x, y, color='black', s=10, marker='o')

    # 检查对数似然值的变化是否小于阈值
    if result[x][y]> need_convergence_threshold:
        print(l0,l1)
        print(f"Convergence reached at iteration {i}. Stopping.")
        break
    print('not enough!',i)
plt.scatter(x, y, color='green', s=50, marker='o',label='end')
print(mcmc_list,flag/10000)

from astropy.io import fits

def make_fits(path,mcmc_list, result):
    # 将数据转换为 NumPy 数组
    num=[i for i in range(len(mcmc_list))]
    num=np.array(num)

    sigmas=[y_labels[i[1]] for i in mcmc_list]
    mu=[x_labels[i[0]] for i in mcmc_list]

    sigmas=np.array(sigmas)
    mu=np.array(mu)

    print('shapes:',num.shape,sigmas.shape,mu.shape)

    # 创建 FITS 表
    col1 = fits.Column(name='index', format='K', array=num)
    col2 = fits.Column(name='sigmas', format='E', array=sigmas)
    col3 = fits.Column(name='mu', format='E', array=mu)
    # col4 = fits.Column(name='dev', format='E', array=dev)
    # col5 = fits.Column(name='signalToNoise', format='E', array=stn)

    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    # 写入 FITS 文件
    tbhdu.writeto(path, overwrite=True)

make_fits('Materials/data/data1.fits',mcmc_list,result)








plt.show()