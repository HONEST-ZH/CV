import cv2 as cv
import numpy as np
import torch
from sklearn import datasets
import torch.utils.data as TorchData
from torchvision import datasets, models, transforms
import time
import glob

TRAIN_NUM = 6
TEST_NUM = 4
def main():
    SVM()
    #Torch()
    return

#######################################
#==============传统方法================#
######################################

#####SVM方法，构造数据集、设置svm模型、使用svm模型#####
def SVM():
    #----------获取训练数据----------------
    trainData, trainLabels, testData, TestLabels = getData()
    # ----------构造svm------------------
    svm = cv.ml.SVM_create()  # 初始化
    svm.setKernel(cv.ml.SVM_LINEAR)  # 设置kernel类型
    svm.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)  # 训练svm
    # ----------使用svm------------------
    result = svm.predict(testData)[1]  # 获取识别标签
    mask = result == TestLabels  # 比较识别结果是否等于实际标签
    correct = np.count_nonzero(mask)  # 计算非零值（相等）的个数
    accuracy = correct * 100.0 / result.size  # 计算准确率（相等个数/全部）
    print("识别准确率为：",accuracy)
    return

#####getData函数，获取训练数据、测试数据及对应标签（预先倾斜矫正和获取HOG值）#####
def getData():
    data=[]   #存储所有数字的所有图像
    for i in range(0,10):
        iTen=glob.glob('data/'+str(i)+'/*.*')   # 所有图像的文件名
        num=[]      #临时列表，每次循环用来存储某一个数字的所有图像
        for number in iTen:    #逐个提取文件名
            # step 1:预处理（读取图像，色彩转换、大小转换）
            image=cv.imread(number)   #逐个读取文件，放入image中
            image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)   #彩色——>灰色
            # x=255-x   #必要时需要做反色处理：前景背景切换
            image=cv.resize(image,(20,20))   #调整大小
            # step2：倾斜校正
            image=deskew(image)   #倾斜校正
            # step3：获取hog值
            hogValue=hog(image)   #获取hog值
            num.append(hogValue)  #把当前图像的hog值放入num中
        data.append(num)  #把单个数字的所有hogvalue放入data，每个数字所有hog值占一行
    x=np.array(data)
    # step4：划分数据集（训练集、测试集）
    trainData=np.float32(x[:,:TRAIN_NUM])
    testData=np.float32(x[:,TRAIN_NUM:])
    # step5：塑形，调整为64列
    trainData=trainData.reshape(-1,64)    #训练图像调整为64列形式
    testData=testData.reshape(-1,64)     #测试图像调整为64列形式
    # step6：打标签
    trainLabels = np.repeat(np.arange(10),TRAIN_NUM)[:,np.newaxis]      #训练图像贴标签
    TestLabels = np.repeat(np.arange(10),TEST_NUM)[:,np.newaxis]       #测试图像贴标签
    return  trainData,trainLabels,testData,TestLabels

#####抗扭斜函数#####
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    s=20
    M = np.float32([[1, skew, -0.5*s*skew], [0, 1, 0]])
    affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
    size=(20,20)   #每个数字的图像的尺寸
    img = cv.warpAffine(img,M,size,flags=affine_flags)
    return img

#####HOG函数#####
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(16*ang/(2*np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(),16) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

#######################################
#============机器学习方法===============#
######################################

def Torch():
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    # 1. 数据准备
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5],std=[0.5])])
    train_dataset = datasets.MNIST(root = "./data/",
                                transform=transform,
                                train = True,
                                download = True)
    test_dataset = datasets.MNIST(root="./data/",
                               transform=transform,
                               train = False)
    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True,
                                                   )
    data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size = BATCH_SIZE,
                                                   shuffle = True)
    # 2. 定义模型
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(stride=2, kernel_size=2))
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(14 * 14 * 128, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(1024, 10))
        def forward(self, x):
            x = self.conv1(x)
            x = x.view(-1, 14 * 14 * 128)
            x = self.dense(x)
            return x
    # 3. 定义优化器损失函数
    model = Model()
    model.to(DEVICE)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # 优化器选用随机梯度下降方式
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss() # 对于多分类一般采用的交叉熵损失函数,
    # 4. 训练模型
    EPOCH=5
    time_open = time.time()
    for t in range(EPOCH):
        training_loss = 0.0
        training_correct = 0
        for step, (input, label) in enumerate(data_loader_train):#一次读一个batch
            #print("batch={}/{}".format(step + 1, int(len(train_dataset) / len(input))))
            input, label = input.to(DEVICE), label.to(DEVICE)
            out = model(input)                 # 输入input,输出out
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
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
        loss= float(training_loss*BATCH_SIZE/len(train_dataset))
        train_Accuracy = 100 * training_correct / len(train_dataset)
        test_Accuracy = 100 * testing_correct / len(test_dataset)
        print("Epoch {:d}/{:d}: Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(t+1,EPOCH,loss,train_Accuracy,test_Accuracy))
    time_end = time.time() - time_open
    print(time_end)
    return

if __name__ == "__main__":
    main()