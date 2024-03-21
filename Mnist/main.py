import torch
import torchvision 
from tqdm import tqdm
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(  # 构建顺序模型，能够自动将层数合并为一个模型
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            # 添加一个2D卷积层，输入通道数为1（例如灰度图），输出通道数为16，卷积核大小为3x3，步长为1，填充为1。
            # 步长为1意味着卷积核每次移动一个像素点，填充为1意味着在输入的每一边上添加了一层零。
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 添加一个最大池化层，池化窗口大小为2x2，步长为2。

            # The size of the picture is 14x14 池化层使得图片大小缩小了
            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(), 
            # 添加一个flatten层，将多维输入一维化，以便在全连接层中使用。
            torch.nn.Linear(in_features = 7 * 7 * 64, out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128, out_features = 10),
            # torch.nn.Softmax(dim=1)
            # 如果在输出层使用了Softmax，然后又用CrossEntropyLoss计算损失，这实际上会导致Softmax被应用两次
            # 这不仅是多余的，还可能导致数值计算上的不稳定性，比如梯度消失问题
        )
    def forward(self, input):
        output = self.model(input)
        return output

BATCH_SIZE = 256
EPOCHS = 10
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean = [0.5], std = [0.5])])  # 这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理

# 构建数据集
# torchvision中的torchvision.datasets库中提供了MNIST数据集的下载地址，因此我们可以直接二调用对应的函数来下载MNIST的训练集和测试集
trainData = torchvision.datasets.MNIST('./data/', train = True, transform = transform, download = True)  # train=True获得训练集
testData = torchvision.datasets.MNIST('./data/', train = False, transform = transform)  # train=True获得数据集
# 使用pytorch的Loader将数据自动打包成迭代器
trainDataLoader = torch.utils.data.DataLoader(dataset = trainData, batch_size = BATCH_SIZE, shuffle = True)
testDataLoader = torch.utils.data.DataLoader(dataset = testData, batch_size = BATCH_SIZE)
net = Net()
print(net.to(device))
# 构建迭代器与损失函数
lossF = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(net.parameters())

history = {'Test Loss':[],'Test Accuracy':[]}
for epoch in range(1,EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit = 'step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)  # 将图像数据移动到指定的设备上（如GPU）
        labels = labels.to(device)  # 同上

        net.zero_grad()  # 初始化梯度
        outputs = net(trainImgs)  # 将图像数据输入网络，获得输出结果
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim = 1)  # 计算预测结果，取输出中最大值的索引作为预测类别
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()  # 执行反向传播，计算每个参数的梯度

        optimizer.step()  # 更新模型参数
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item()))  # 设置进度条显示当前epoch、损失和准确率
        
        if step == len(processBar) - 1:  # 检查是否为当前epoch的最后一步
            correct, totalLoss = 0, 0
            net.train(False)  # 将模型设置为测试模式，禁用梯度计算
            with torch.no_grad():
                for testImgs,labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs,labels)
                    predictions = torch.argmax(outputs,dim = 1)
                    
                    totalLoss += loss  # 累加计算损失
                    correct += torch.sum(predictions == labels)  # 累加计算正确预测数量
                    
                    testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                    testLoss = totalLoss / len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()

plt.plot(history['Test Loss'], label = 'Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history['Test Accuracy'], color = 'red', label = 'Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

torch.save(net,'./model.pth')