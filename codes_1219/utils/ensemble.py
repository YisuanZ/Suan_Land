import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import os, sys
import joblib
from sklearn.cluster import KMeans
from scipy.stats import norm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

from gpro.utils.base import write_seq, write_exp
from gpro.utils.utils_predictor import seq2onehot, open_fa, open_exp
from gpro.predictor.attnbilstm.attnbilstm import SequenceData, TestData


'''
Model 1: NN + GP Layer
Ensemble Model: basic, 3 Linear layer + Gaussian + ReLU
'''

class GaussianLayer(nn.Module):
    def __init__(self, output_dim):
        super(GaussianLayer, self).__init__()
        self.output_dim = output_dim
        self.w_1 = nn.Parameter(torch.Tensor(30, self.output_dim).normal_())
        self.w_2 = nn.Parameter(torch.Tensor(30, self.output_dim).normal_())
        self.b_1 = nn.Parameter(torch.Tensor(self.output_dim).normal_())
        self.b_2 = nn.Parameter(torch.Tensor(self.output_dim).normal_())
        
    def forward(self, x):
        output_mu  = torch.matmul(x, self.w_1) + self.b_1  # output mu
        output_sig = torch.matmul(x, self.w_2) + self.b_2  # output sigma
        output_sig_pos = torch.log(1 + torch.exp(output_sig)) + 1e-06  # always being positive
        return [output_mu, output_sig_pos]


class EnsembleModel(nn.Module):
    def __init__(self, n_tokens = 4, latent_dim = 128, seq_len = 80):
        super(EnsembleModel, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_tokens * seq_len, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 30),
            GaussianLayer(1)
        )
    def forward(self, x):
        mu, sigma = self.model(x)
        return [mu, sigma]

def custom_loss(y_true, mu, sigma):
    return torch.mean(0.5 * torch.log(sigma) + 0.5 * torch.div(torch.square(y_true - mu), sigma)) + 1e-6

'''
start_ensembling: Ensembling a model for validation
inputs: 
(1) seqs, expr: for training models
(2) model: basic model class
(3) prefix: saving prefix for checkpoints, e.g., "./checkpoints/DeepEnsembles"
(4) tag: saving tag
(5) epochs: training epochs for each model
(6) size: default samping size for each model
'''

def prefix_check(prefix):
    if prefix.endswith("/"):
        prefix = prefix[:-1]
    if not os.path.exists(prefix + "/supps"):
        os.makedirs(prefix + "/supps")
    if not os.path.exists(prefix + "/docs"):
        os.makedirs(prefix + "/docs")
    if not os.path.exists(prefix + "/tables"):
        os.makedirs(prefix + "/tables")
    if not os.path.exists(prefix + "/checks"):
        os.makedirs(prefix + "/checks")
    return prefix

def model_reinit(model):
    seq_len = model.seq_len
    model.__init__(seq_len = seq_len)
    return model

def start_ensembling(seqs, expr, model, prefix, tags, epochs=100, size=5000):
    
    cnt = 1
    for tag in tags:
        random.seed(cnt)
        cnt += 1
        idx = random.sample(range(len(seqs)), k=size)
        seqs = np.array(seqs)[idx]
        expr = np.array(expr)[idx]

        feature = seq2onehot(seqs, 80)
        feature = torch.tensor(feature, dtype=float)
        label = torch.tensor(expr, dtype=float)
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]

        r = int(len(seqs) * 0.8)
        train_seqs = seqs[0:r]
        train_dataset = SequenceData(feature[0:r], label[0:r])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        valid_seqs = seqs[r:]
        valid_dataset = SequenceData(feature[r:], label[r:])
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

        prefix = prefix_check(prefix)

        write_seq("{}/supps/seq_{}.txt".format(prefix, tag), seqs[0:r])
        write_exp("{}/supps/exp_{}.txt".format(prefix, tag), expr[0:r])

        model = model_reinit(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = custom_loss

        train_log_filename = "{}/docs/log_{}.txt".format(prefix, tag)
        train_csv_filename = "{}/tables/log_{}.csv".format(prefix, tag)
        train_model_filename = "{}/checks/checkpoint_{}.pth".format(prefix, tag)


        for epoch in tqdm(range(0,epochs)):
            model.train()
            train_epoch_loss = []
            for idx,(feature,label) in enumerate(train_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                mu, sigma = model(feature)
                optimizer.zero_grad()
                loss = criterion(label.float(), mu.flatten(), sigma.flatten())
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

            model.eval()
            valid_expr = []
            valid_mu = []
            valid_sigma = []
            for idx,(feature,label) in enumerate(valid_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                mu, sigma = model(feature)
                valid_expr += label.float().tolist()
                valid_mu += mu.flatten().tolist()
                valid_sigma += sigma.flatten().tolist()
            coefs = np.corrcoef(valid_expr,valid_mu)
            coefs = coefs[0, 1]
            valid_coefs = coefs

            print("real expression samples: ", valid_expr[0:5])
            print("pred expression samples: ", valid_mu[0:5])
            print("current coeffs: ", valid_coefs)
            cor_pearsonr = pearsonr(valid_expr, valid_mu)
            print("current pearsons: ",cor_pearsonr)

            if (epoch%10 == 0):
                to_write = "epoch={}, loss={}\n".format(epoch, np.average(train_epoch_loss))
                with open(train_log_filename, "a") as f:
                    f.write(to_write)
                df_valid = pd.DataFrame({"seqs": valid_seqs, "expr": valid_expr, "mu": valid_mu, "sigma": valid_sigma})
                df_valid.to_csv(train_csv_filename)
            if (epoch%20 == 0):
                torch.save(model.state_dict(), train_model_filename)
    return

'''
get_ensembles: get ensembling for new samples from available ensembings
input: 
(1) seqs: for training models
(2) model: 
(3) prefix: for path_check, ./checkpoints/DeepEnsembles
(4) tags: a list for tag, containing all tags to be validated
output:
mu, sigma for further validation
'''


def get_ensembles(seqs, model, prefix, tags):
    test_seqs = seqs
    test_feature = seq2onehot(test_seqs, 80)  # 转化成独热编码格式
    test_feature = torch.tensor(test_feature, dtype=float)
    test_dataset = TestData(test_feature)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]

    model = model.to(device)
    prefix = prefix_check(prefix)
    
    test_mu_list, test_sigma_list = [], []  # 初始化两个列表来存储所有标签下的均值和标准差预测结果
    for tag in tags:
        test_mu, test_sigma = [], []
        path_check = "{}/checks/checkpoint_{}.pth".format(prefix, tag)  # 生成模型检查点的路径
        model.load_state_dict(torch.load(path_check))  # 加载模型状态
        model.eval()  # 将模型设置为评估模式
        # 迭代数据加载器并进行预测
        for idx, feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)  # 将特征转换为适当的数据类型和设备，并调整维度
            mu, sigma = model(feature)  # 对每个批次的数据进行预测，获取均值和标准差
            test_mu += mu.flatten().tolist()  # 将预测结果添加到列表中
            test_sigma += sigma.flatten().tolist()

        test_mu_list.append(test_mu)
        test_sigma_list.append(test_sigma)
    
    test_mu = np.mean(test_mu_list, axis=0)
    test_sigma = np.sqrt(np.mean(test_sigma_list + np.square(test_mu_list), axis=0) - np.square(test_mu) )
    return test_mu, test_sigma


def plot_diagram(expr, mu, sigma, savepath):
    idx = sorted(range(len(expr)), key=lambda k: expr[k], reverse=False)
    expr = np.array(expr)[idx]
    test_mu = np.array(mu)[idx]
    test_sigma = np.array(sigma)[idx]

    index = list(range(len(expr)))
    sns.set_style("darkgrid")
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (8,6), dpi = 400)
    plt.plot([i for i in index], [i for i in expr], 'tab:orange', linewidth=2, label="real")
    plt.plot([i for i in index], [i for i in test_mu], 'tab:blue', linewidth=2, label="mu")

    upper = [i+k for i,k in zip(test_mu, test_sigma)]
    lower = [i-k for i,k in zip(test_mu, test_sigma)]
    plt.fill_between([i for i in index], lower, upper, color='cornflowerblue', alpha=0.6, label="sigma")
    
    ax.set_xlabel('Index', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend()
    plt.show()
    plt.savefig(savepath)


'''
get_neighbors: get the nearest k-samples for making current regression (contribution)
inputs: 
(1) seqs, expr: for training models
(2) model: basic model class
(3) prefix: saving prefix for checkpoints, e.g., "./checkpoints/DeepEnsembles"
(4) tag: saving tag
(5) report: whether output the report to txt file
(6) report_num: number of k
(7) report_path: the output path for reports
'''

fmap_block = dict()
def forward_hook(module, input, output):
    fmap_block['input'] = input
    fmap_block['output'] = output

def feature_embedding(seqs, model):
    feature = seq2onehot(seqs, 80)
    feature = torch.tensor(feature, dtype=float)
    dataset = TestData(feature)
    dataloader = DataLoader(dataset=dataset, batch_size = 64, shuffle=False)
    device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
    
    feature_list = []
    output_list = []
    for idx,feature in enumerate(dataloader,0):
        input = feature.to(torch.float32).to(device).permute(0,2,1)
        mu, sigma = model(input)
        feature_list += fmap_block['output'].tolist()   
        output_list += mu.flatten().tolist()
    return feature_list, output_list


# 找出给定序列在训练数据集中的“邻居”，即与这些序列最相似的训练样本。它通过比较特征空间中的距离来实现。
def get_neighbors(seqs, expr, model, prefix, tags, report=True, report_num=5, report_path=None):
    
    prefix = prefix_check(prefix)
    
    for tag in tags:
        path_check = "{}/checks/checkpoint_{}.pth".format(prefix, tag)
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]

        model = model_reinit(model)
        model = model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()

        for (name, module) in model.named_modules():
            if name == "model":
                module[-2].register_forward_hook(hook=forward_hook)


        emb, pred = feature_embedding(seqs, model)
        test_seqs = seqs
        test_pred = pred
        test_expr = expr

        train_seqs = open_fa("{}/supps/seq_{}.txt".format(prefix, tag))
        train_exps = open_exp("{}/supps/exp_{}.txt".format(prefix, tag), operator="direct")

        supp_sample = []
        supp_distance = []
        supp_expr = []
        supp_pred = []


        for i in tqdm(range(len(train_seqs))):
            train_seq = train_seqs[i]
            train_exp = train_exps[i]
            train_feature, train_pred = feature_embedding([train_seq], model)

            supp_sample.append(train_seq)
            supp_expr.append(train_exp)
            supp_pred.append(train_pred[0])
            train_feature = (np.array(train_feature)).T
            emb_norm = np.linalg.norm(emb)
            train_feature_norm = np.linalg.norm(train_feature)
            supp_distance.append(np.dot(emb / emb_norm, train_feature / train_feature_norm))

        supp_distance = [item[0][0] for item in supp_distance]

        idx = sorted(range(len(supp_distance)), key=lambda k: supp_distance[k], reverse=True)
        supp_sample = np.array(supp_sample)[idx]
        supp_distance = np.array(supp_distance)[idx]
        supp_expr = np.array(supp_expr)[idx]
        supp_pred = np.array(supp_pred)[idx]

        if(report):
            if(report_path):
                sys.stdout = open(report_path, "a")

            K = report_num
            print("#" + "tag{} ".format(tag) + "-----" * 20 + "#")
            print("current seqs for validation: ", test_seqs)
            print("predicted value for current seqs: ", test_pred)
            print("real expression for current seqs: ", test_expr)
            print("nearest samples in the training dataset: \n", supp_sample[0:K])
            print("predicted value for nearest samples: \n", supp_pred[0:K])
            print("real expression for nearest samples: \n", supp_expr[0:K])
            print("cosine similarity with neareast samples: \n", supp_distance[0:K])

            sys.stdout = sys.__stdout__
    
        
    return

'''
validation: plot boxplot
'''

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

'''
criterion: sampling strategy 
(1) PI (probability Improvement)
(2) UCB (Upper Confidence Boundary)
'''

