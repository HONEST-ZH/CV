
# version:python 3.7.9      pytorch :1.7.0
# function:利用神经网络进行鸢尾花分类

import numpy as np
import torch
from collections import Counter
from sklearn import datasets
import torch.nn.functional as Fun

import torch.utils.data as TorchData
SPLIT_NUM = 120
BATCH_SIZE = SPLIT_NUM
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
# 1. 数据准备
dataset = datasets.load_iris()
data = dataset['data']
labels = dataset['target']

labels = labels.reshape(-1, 1)
total_data = np.hstack((data, labels))
np.random.shuffle(total_data)
data_train = total_data[0:SPLIT_NUM, :-1]
data_test = total_data[SPLIT_NUM:, :-1]

label_train = total_data[0:SPLIT_NUM, -1]
label_test = total_data[SPLIT_NUM:, -1]

train_dataset = TorchData.TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(label_train).long())
test_dataset = TorchData.TensorDataset(torch.from_numpy(data_test).float(), torch.from_numpy(label_test).long())

data_loader_train = TorchData.DataLoader(dataset=train_dataset,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                               )

data_loader_test = TorchData.DataLoader(dataset=test_dataset,
                                               batch_size = BATCH_SIZE,
                                               shuffle = True)
# 2. 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 定义隐藏层网络
        self.out = torch.nn.Linear(n_hidden, n_output)   # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hidden(x))      # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
        x = self.out(x)                   # 输出层不用激活函数
        return x

# 3. 定义优化器损失函数
net = Net(n_feature=4, n_hidden=20, n_output=3).to(DEVICE)    #n_feature:输入的特征维度,n_hiddenb:神经元个数,n_output:输出的类别个数
#optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # 优化器选用随机梯度下降方式
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss() # 对于多分类一般采用的交叉熵损失函数,

# 4. 训练模型
EPOCH=80
for t in range(EPOCH):
    training_loss = 0.0
    training_correct = 0
    for step, (input, label) in enumerate(data_loader_train):#一次读一个batch
        input, label = input.to(DEVICE), label.to(DEVICE)
        out = net(input)                 # 输入input,输出out
        _, pred = torch.max(out.data, 1)
        loss = loss_func(out, label)     # 输出与label对比
        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 前馈操作
        optimizer.step()        # 使用梯度优化器

        training_loss += loss.data
        training_correct += torch.sum(pred == label.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        outputs = net(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    loss= float(training_loss*BATCH_SIZE/len(data_train))
    train_Accuracy = 100 * training_correct / len(data_train)
    test_Accuracy = 100 * testing_correct / len(data_test)
    print("Epoch {:d}/{:d}: Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(t+1,EPOCH,loss,train_Accuracy,test_Accuracy))
