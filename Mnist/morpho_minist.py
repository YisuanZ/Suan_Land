import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 定义一个多层感知器（前馈神经网络）
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        构建前馈神经网络模型
        :param input_dim: 输入维度
        :param hidden_dim1: 隐藏层1的维度
        :param hidden_dim2: 隐藏层2的维度
        :param output_dim: 输出层的维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

# 读取训练集
train_X = np.load('./data/train/train_minist.npy')  # 数字矩阵
train_label = pd.read_csv('./data/train/train_label.csv')
train_number = train_label['number']  # 数字标签
train_size = train_label['size']  # 粗细标签

# 读取测试集
test_X = np.load('./data/test/test_minist.npy')
test_label = pd.read_csv('./data/test/test_label.csv')
test_number = test_label['number']
test_size = test_label['size']

# 查看数据集规模
print(f"size of train: {train_X.shape}, size of test: {test_X.shape}")

# ----------------------------->第一题（必做）
# TODO 1:使用Logistic回归拟合训练集的X数据和size标签,并对测试集进行预测

train_X_reshape = train_X.reshape(39980, 784)
print(train_X_reshape.shape)
print(test_size.shape)

model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='multinomial',
          n_jobs=None, penalty='l2', random_state=0, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False)
model.fit(train_X_reshape, train_size)

test_X_reshape = test_X.reshape(6693, 784)

test_size_pred = model.predict(test_X_reshape)
test_size_probas = model.predict_proba(test_X_reshape)
print("accuracy score:", accuracy_score(test_size, test_size_pred))
print("precison score:", precision_score(test_size, test_size_pred))
print("recall_score:", recall_score(test_size, test_size_pred))
print("f1_score:", f1_score(test_size, test_size_pred))
print("auROC:", roc_auc_score(test_size, test_size_probas[:, 1]))

fpr, tpr, thresholds = roc_curve(test_size, test_size_probas[:, 1])
plt.plot(fpr, tpr)
plt.show()

# # ---------------------------->第二题（必做）
# # TODO 2:使用Softmax回归拟合训练集的X数据和number标签,并对测试集进行预测

train_X_reshape = train_X.reshape(39980, 784)
print(train_X_reshape.shape)
print(test_number.shape)
test_X_reshape = test_X.reshape(6693, 784)

model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='multinomial',
          n_jobs=None, penalty='l2', random_state=0, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False)
model.fit(train_X_reshape, train_number)

test_number_pred = model.predict(test_X_reshape)
test_number_probas = model.predict_proba(test_X_reshape)

print("accuracy score:", accuracy_score(test_number, test_number_pred))
print("macro-precison score:", precision_score(test_number, test_number_pred, average = "macro"))
print("macro-recall_score:", recall_score(test_number, test_number_pred, average = "macro"))
print("macro-f1_score:", f1_score(test_number, test_number_pred, average = "macro"))
# print(test_number.shape, test_number_pred.shape)
# print(test_number)
# print(test_number_pred)
# test_number_one_hot = label_binarize(test_number, np.arange(10))
test_number_one_hot = np.zeros((6693, 10), dtype = int)
for i in range(6693):
    test_number_one_hot[i][test_number[i]] = 1
print("auROC:", roc_auc_score(test_number_one_hot, test_number_probas, average = "micro"))

print(confusion_matrix(test_number, test_number_pred))

# ---------------------------->第三题（选做）
# 转换NumPy数组为PyTorch张量

train_X = torch.tensor(train_X, dtype=torch.float32).reshape(-1, 28 * 28)
test_X = torch.tensor(test_X, dtype=torch.float32).reshape(-1, 28 * 28)
train_number = torch.tensor(train_number)
test_number = torch.tensor(test_number)

# 将数据和标签合并成一个数据集
train_dataset = torch.utils.data.TensorDataset(train_X, train_number)
test_dataset = torch.utils.data.TensorDataset(test_X, test_number)

# 创建数据加载器，每次迭代从数据集中随机选择一个批次
batch_size = 512
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型参数
input_dim = 28 * 28
hidden_dim1 = 128
hidden_dim2 = 128
output_dim = 10  # 类别数目（0到9）
# 初始化模型
model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
# 使用交叉熵损失
criterion = nn.CrossEntropyLoss()
# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型的轮次
num_epochs = 500
# 每轮训练的损失函数的均值组成的列表
losses_list = []
# TODO 3：完成前馈神经网络的分批次训练和损失函数的记录

for epoch in range(num_epochs):

    runningloss = 0.0
    for batch_times, data in enumerate(train_loader, 0):
        picture, train_num = data
       
        optimizer.zero_grad()

        # print(batch_times)

        train_num_pred = model(picture)
        loss = criterion(train_num_pred, train_num)
        loss.backward()
        optimizer.step()
                
        runningloss += loss.item()
        if batch_times % 79 == 78:
            print(epoch+1, runningloss/79)
            losses_list.append(runningloss/79)
            runningloss = 0.0

print(losses_list)

# 绘制损失函数变化图
plt.plot(losses_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train loss varies with epoch')
plt.show()

# 在测试集上进行预测
model.eval()
with torch.no_grad():
    test_pred = model(test_X)
    _, predicted = torch.max(test_pred, 1)  # 选择具有最高概率的类别作为预测类别

#效果评估
print("accuracy score:", accuracy_score(test_number, predicted))
print("macro-precison score:", precision_score(test_number, predicted, average = "macro"))
print("macro-recall_score:", recall_score(test_number, predicted, average = "macro"))
print("macro-f1_score:", f1_score(test_number, predicted, average = "macro"))

test_number_one_hot = np.zeros((6693, 10), dtype = int)
for i in range(6693):
    test_number_one_hot[i][test_number[i]] = 1
print("auROC:", roc_auc_score(test_number_one_hot, test_pred, average = "micro"))

print(confusion_matrix(test_number, predicted))
