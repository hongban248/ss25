
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

