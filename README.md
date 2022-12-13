# 肌电信号处理大作业

本仓库为2022-2023学年第一学期《健康与医学信息处理》期末课程作业代码

| 姓名（网上冲浪不实名~） | 学号    |
| ----------------------- | ------- |
| lyj                     | 1951578 |
| pw                      | 1952734 |
| wjb                     | 1951962 |
| yzk                     | 1952105 |

#### 依赖环境

h5py==3.7.0

matplotlib==3.6.2

numpy==1.23.4

scipy==1.9.3

torch==1.13.0

torchvision==0.14.0

torchsummary

#### 文件目录

`scripts` 存放pytorch脚本代码

`MATLAB` MATLAB预处理脚本

`feature` 特征向量数据集，训练使用

`model` 训练模型的参数

#### 代码脚本

```bash
ninapro_utils.py				# 数据集处理工具函数
sEMG_proc_CNNs.ipynb			# 卷积神经网络方法分类
sEMG_proc_dataExtraction.ipynb	# 数据可视化与处理脚本
sEMG_proc_DWT.ipynb				# 小波变换脚本
sEMG_proc_LFM.ipynb				# 低秩多模态融合脚本
semg_train.ipynb				# 其他方法训练脚本
```

#### 关于数据

数据过大，故存网盘了。可以下载到对应目录使用。