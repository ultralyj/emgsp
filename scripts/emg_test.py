# -*- coding: utf-8 -*-
# file name: ninapro_utils
# author: ultralyj
# e-mail: 1951578@tongji.edu.cn
# version: v1.0
# date: 2022-12-3
# brief: only for test

import scipy.io
import numpy as np
from ninapro_utils import *
import matplotlib as mpl 
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
import math

# 数据集的地址
db7_path = 'E:/db7/'


# iSampleRate = 2000  # 采样频率
# colors=list(mcolors.TABLEAU_COLORS.keys()) #颜色变化
# for j in range(1, 2):
#     h5 = h5py.File('F:/DB2/refilter/DB2_s' + str(j) + 'refilter.h5', 'r')
#     alldata = h5['alldata'][:]
#     seglist = segment(alldata, 1, 12)
#     bnlist = bnsegment(seglist)
#     iemg = bnlist[1].data
#     iSampleCount = iemg.shape[0]  # 采样数
#     plt.figure(figsize=(20,8))
#     t = np.linspace(0, iSampleCount / iSampleRate, iSampleCount)
#     for i in range(12):
#         plt.subplot(12,1,i+1)
#         plt.plot(iemg[10000:12000,i],color=mcolors.TABLEAU_COLORS[colors[int(math.fabs(i-2))]])
#         plt.axis('off') 
#     plt.show()
if __name__ == "__main__":
    s1_e1_raw = get_ninapro(db7_path + 'Subject_1/','S1_E1_A1.mat')
    
    # 获取动作2的范围
    s2_range = find_sample_range(s1_e1_raw,2)
    data = s1_e1_raw['emg'][s2_range[0]:s2_range[1],:]
    # 颜色变化
    colors=list(mcolors.TABLEAU_COLORS.keys()) 
    plt.plot(data,linewidth = 0.1, alpha=0.5)
    plt.show()
    print('---finish---') 