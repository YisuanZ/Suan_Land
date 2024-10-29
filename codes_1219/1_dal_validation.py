'''
本文件验证主动学习的可行性
'''

'''
Step1:
将文档涉及的所有函数封装到utils中以调用 并且为每个函数增加对应的prefix/tag 等接口 以及允许ensemble不同的模型

Step2:
接下来我们建立一个新文档评估我们的主动学习方法的合理性
数据源: S, 3500000条带重复样本
/home/qxdu/gpro/gpro/gpro/demos/test_3_evolutionary/training_data_Glu.txt
/home/qxdu/gpro/gpro/test/ai_paper_1126/aviv/2_plot_uncertainty.py 记录了如何处理这些样本

1. 从数据源S生成一组样本B 用于测试 B1代表[预测高表达 实际高表达]的样本 B2代表[预测高表达 实际低表达]的样本
2. 用数据源S剩下的样本构建两个主动学习集合 一个代表采集函数最偏好的样本A1, 一个代表随机采样的样本A2 进行若干轮(1轮)迭代
3. 验证经过A1学习的模型效果好于经A2学习的模型
'''

import os, sys
import random
import numpy as np
import pandas as pd

from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

sys.path.append('/home/qxdu/gpro/gpro/test/ai_paper_1217')
from utils.ensemble import EnsembleModel, start_ensembling
from utils.ensemble import get_ensembles, plot_diagram
from utils.ensemble import get_neighbors
from gpro.utils.utils_predictor import seq2onehot, open_fa, open_exp

print("test")
df = pd.read_csv("/home/qxdu/gpro/gpro/test/ai_paper_1126/aviv/dataset/bins/GSE_atLeast100Counts.txt", sep='\t', header=None, names=['seqs', 'expr'])
df = df[df['seqs'].apply(len) == 110]  # 过滤出 'seqs' 列中长度为 110 的条目
seqs = list(df.loc[:,"seqs"].str.slice(17, -13))  # 从每个序列中切割出一部分，保留从第 18 个字符到倒数第 13 个字符之间的片段
expr = list(df.loc[:,"expr"])  # 将 'expr' 列的所有值转换成一个列表，用于后续处理

# 1. 重新训练集成模型
model = EnsembleModel()  
tags = ["r1n{}".format(i+1) for i in range(10)]  # 这里创建了一个列表 tags，包含了10个标签，分别为 r1n1 到 r1n10
                                                 # TODO 这个tags是什么东西？一个tag是指一个特定的模型吗？是的，就是模型编号
start_ensembling(seqs, expr, model, prefix="./checkpoints/DeepAL_Aviv/R1", tags=tags, epochs=100, size=1000) # 参数定义了保存训练过程中生成的模型检查点的路径


# 2. 评估和绘图
random.seed(42)
idx = random.sample(range(len(seqs)), k=100)  # 从序列数据中随机选择100个不重复的索引
seqs = np.array(seqs)[idx]  # 根据这些随机索引选择对应的序列和表达式数据
expr = np.array(expr)[idx]

model = EnsembleModel()
tags = ["r1n{}".format(i+1) for i in range(10)]

test_mu, test_sigma = get_ensembles(seqs, model, prefix="./checkpoints/DeepAL_Aviv/R1", tags=tags)
plot_diagram(expr, test_mu, test_sigma, savepath="./test.png")

# 3. 获得邻居

seqs = open_fa("/home/qxdu/gpro/gpro/test/ai_paper_1126/aviv/dataset/bins/test_seq.txt")
expr = open_exp("/home/qxdu/gpro/gpro/test/ai_paper_1126/aviv/dataset/bins/test_exp.txt", operator="direct")
model = EnsembleModel()
prefix="./checkpoints/DeepAL_Aviv/R1"
tags = ["r1n{}".format(i+1) for i in range(10)]

mu, sigma = get_ensembles(seqs, model, prefix, tags)
# 根据标准差 sigma 对样本进行降序排序。这意味着首先考虑那些模型预测不确定性最高的样本。
idx = sorted(range(len(sigma)), key=lambda k: sigma[k], reverse=True)
seqs = np.array(seqs)[idx]
expr = np.array(expr)[idx]
mu = np.array(mu)[idx]
sigma = np.array(sigma)[idx]
# 按照不确定性对生鲜的元素也进行重排序

print("mu, sigma, expr = ", mu[0], sigma[0], expr[0])
get_neighbors([seqs[0]], [expr[0]], model, prefix=prefix, tags=tags,report_path="./reports/R1_max.txt")  # 找出最不确定的数据以及与之相近的训练样本
    
print("mu, sigma, expr = ", mu[-2], sigma[-2], expr[-2])
get_neighbors([seqs[-2]], [expr[-2]], model, prefix=prefix, tags=tags,report_path="./reports/R1_min.txt")  # 找出最确定的数据以及与之相近的训练样本

'''
相当一部分所谓"最确定"的样本表现非常糟糕 这需要我们警惕采样函数 
sigma低的样本未必好, 这是一个复杂的权衡机制 一开始设置成1或许不错, 每次将新的样本直接合并到原始训练集中
下面采样的代码等会记得封装
'''

# 4. 采样测试  贝叶斯优化
# 这个函数用于去除重复的序列，同时保持序列和其对应表达值的关系。它返回两个列表，分别包含唯一的序列和相应的表达值。
def unique_list(seq, exp):  
    unique_dict = {}
    unique_seq = []
    unique_exp = []
    for item, value in zip(seq, exp):
        if item not in unique_dict:
            unique_dict[item] = value
            unique_seq.append(item)
            unique_exp.append(value)
    return unique_seq, unique_exp


def plot_boxplot(plot_path, barplot_data):
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

    ax = sns.boxplot( x="label", y="expression",  data=barplot_data,  boxprops=dict(alpha=.9), # hue="label", hue_order = hue_order,
                      fliersize=1, flierprops={"marker": 'x'}, palette="viridis_r") # # palette="viridis_r"
    h,_ = ax.get_legend_handles_labels()

    ax.set_xlabel('Acquisition Functions', fontsize=10)
    ax.set_ylabel('Expressions', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.show()
    plt.savefig(plot_path)

seqs = open_fa("/home/qxdu/gpro/gpro/test/ai_paper_1126/aviv/dataset/bins/test_seq.txt")
expr = open_exp("/home/qxdu/gpro/gpro/test/ai_paper_1126/aviv/dataset/bins/test_exp.txt", operator="direct")
seqs, expr = unique_list(seqs, expr)
model = EnsembleModel()
prefix="./checkpoints/DeepAL_Aviv/R1"
tags = ["r1n{}".format(i+1) for i in range(10)]
mu, sigma = get_ensembles(seqs, model, prefix, tags)

eps1 = 0.01
eps2 = 0.5
crit = norm.cdf( ( (mu - 18) * eps1 - eps2)/sigma ) # 可以筛选crit>0.5的样本点
# 高 crit 值意味着一个序列有较高的概率表现出高表达量，因此这样的序列可能更值得关注或进一步研究
idx = sorted(range(len(crit)), key=lambda k: crit[k], reverse=True)
seqs = np.array(seqs)[idx]
expr = np.array(expr)[idx]
mu = np.array(mu)[idx]
sigma = np.array(sigma)[idx]

N = 100
df_max = pd.DataFrame({"seq": seqs[0:N], "expr": expr[0:N], "mu": mu[0:N], "sigma": sigma[0:N]})
df_max.to_csv("./reports/R1_max.csv")
df_min = pd.DataFrame({"seq": seqs[-N:], "expr": expr[-N:], "mu": mu[-N:], "sigma": sigma[-N:]})
df_min.to_csv("./reports/R1_min.csv")

df_concat = pd.DataFrame({"expression": list(expr[0:N]) + list(expr[-N:]), "label": ["max"] * N + ["min"] * N})
plot_boxplot("./reports/R1_boxplot.png", df_concat)

# 整个过程用于分析和可视化模型对序列的预测表达量，
# 通过比较最高和最低预测表达量的序列集合，来评估模型的预测能力和特定序列的表达潜力。
# 这样的分析对于理解模型的行为以及指导未来的实验设计和决策非常有帮助。