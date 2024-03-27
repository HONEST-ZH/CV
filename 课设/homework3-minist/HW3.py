import numpy as np
import cv2 as cv
import torch #用于深度学习的库Pytorch
from torchvision import datasets #torchvision用于处理图像和视频数据，datasets中有常见的数据集
from torchvision.transforms import v2 as transforms2 #transforms是对数据集的处理和变换
import glob #获取全部的文件名
import os
import random
from matplotlib import pyplot as plt
#全局变量参数
TOTAL_NUM = 10#数据集大小
TRAIN_NUM = 1 #训练集大小
TEST_NUM = TOTAL_NUM - TRAIN_NUM #测试集大小
DESKEW_FLAG = 1#是否进行倾斜矫正的标志位
DATESET_FLAG = 0#数据集标志：0，small_data;1,minist_data；2,affine_data
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    #=====从MINIST数据集或者进行仿射变换获取图像数据，已存放在data中，无需再次执行=====#
    #get_minist_data()
    #affine_data()

    # SVM方法
    # 引入控制SVM训练的全局变量，便于绘图
    global TOTAL_NUM
    global TRAIN_NUM
    global TEST_NUM
    global DESKEW_FLAG
    global DATESET_FLAG
    #=====开启倾斜矫正，测试不同训练集，不同数据规模下传统方法的准确率=====#
    DESKEW_FLAG = 1
    small_accuracy_deskew = []
    minist_accuracy_deskew = []
    affine_accuracy_deskew = []
    #-----small_data中开启倾斜矫正，不同数据规模下的准确率-----#
    DATESET_FLAG = 0
    TOTAL_NUM = 10
    for i in range(1,10):
        TRAIN_NUM = i
        TEST_NUM = TOTAL_NUM - TRAIN_NUM
        accuracy = SVM()
        small_accuracy_deskew.append(accuracy)
    #-----minist_data中开启倾斜矫正，不同数据规模下的准确率-----#
    DATESET_FLAG = 1
    TOTAL_NUM = 100
    for i in range(1, 100):
        TRAIN_NUM = i
        TEST_NUM = TOTAL_NUM - TRAIN_NUM
        accuracy = SVM()
        minist_accuracy_deskew.append(accuracy)
    #-----affine_data中开启倾斜矫正，不同数据规模下的准确率-----#
    DATESET_FLAG = 2
    TOTAL_NUM = 100
    for i in range(1, 100):
        TRAIN_NUM = i
        TEST_NUM = TOTAL_NUM - TRAIN_NUM
        accuracy = SVM()
        affine_accuracy_deskew.append(accuracy)
    #=====关闭倾斜矫正，测试不同训练集，不同数据规模下传统方法的准确率=====#
    DESKEW_FLAG = 0
    small_accuracy = []
    minist_accuracy = []
    affine_accuracy = []
    #-----small_data中关闭倾斜矫正，不同数据规模下的准确率-----#
    DATESET_FLAG = 0
    TOTAL_NUM = 10
    for i in range(1, 10):
        TRAIN_NUM = i
        TEST_NUM = TOTAL_NUM - TRAIN_NUM
        accuracy = SVM()
        small_accuracy.append(accuracy)
    #-----minist_data中关闭倾斜矫正，不同数据规模下的准确率-----#
    DATESET_FLAG = 1
    TOTAL_NUM = 100
    for i in range(1, 100):
        TRAIN_NUM = i
        TEST_NUM = TOTAL_NUM - TRAIN_NUM
        accuracy = SVM()
        minist_accuracy.append(accuracy)
    #-----affine_data中关闭倾斜矫正，不同数据规模下的准确率-----#
    DATESET_FLAG = 2
    TOTAL_NUM = 100
    for i in range(1, 100):
        TRAIN_NUM = i
        TEST_NUM = TOTAL_NUM - TRAIN_NUM
        accuracy = SVM()
        affine_accuracy.append(accuracy)
    #=====绘图，展示准确率差异=====#
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    fig, ax = plt.subplots()# 使用标签绘制每个列表
    ax.plot(small_accuracy_deskew, label='原始数据开启倾斜矫正，不同数据规模下的准确率')
    ax.plot(minist_accuracy_deskew, label='minist数据开启倾斜矫正，不同数据规模下的准确率')
    ax.plot(affine_accuracy_deskew, label='仿射变换数据开启倾斜矫正，不同数据规模下的准确率')
    ax.plot(small_accuracy, label='原始数据关闭倾斜矫正，不同数据规模下的准确率')
    ax.plot(minist_accuracy, label='minist数据关闭倾斜矫正，不同数据规模下的准确率')
    ax.plot(affine_accuracy, label='仿射变换数据关闭倾斜矫正，不同数据规模下的准确率')
    ax.legend()# 向图表添加图例
    ax.set_title('不同数据集是否使用倾斜校正的准确率对比')# 设置标题和标签
    ax.set_xlabel('数据量')
    ax.set_ylabel('准确率')
    fig.savefig("accuracy.png")

    #深度学习方法
    DESKEW_FLAG = 0
    Train_Accuracy, Test_Accuracy = Torch()
    DESKEW_FLAG = 1
    Train_Accuracy2, Test_Accuracy2 = Torch()
    # =====绘图，展示准确率差异=====#
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    fig1, ax1 = plt.subplots()  # 使用标签绘制每个列表
    ax1.plot(Train_Accuracy, label='未倾斜校正，深度学习方法在训练集上的准确率')
    ax1.plot(Test_Accuracy, label='未倾斜校正，深度学习方法在测试集上的准确率')
    ax1.plot(Train_Accuracy2, label='倾斜校正，深度学习方法在训练集上的准确率')
    ax1.plot(Test_Accuracy2, label='倾斜校正，深度学习方法在测试集的准确率')
    ax1.legend()  # 向图表添加图例
    ax1.set_title('深度学习下是否使用倾斜校正的准确率对比')  # 设置标题和标签
    ax1.set_xlabel('EPOCH')
    ax1.set_ylabel('准确率')
    fig1.savefig("CNN.png")
    return
#######################################
#============数据集处理方式==============#
######################################

#=======从MINIST数据集获得图片以便于传统方法识别======#
def get_minist_data():
    # 加载Mnist数据集
    transform = transforms2.Compose([transforms2.ToTensor(),
                                    transforms2.Normalize(mean=[0.5], std=[0.5])])
    mnist_train = datasets.MNIST(root="./data/",
                                 transform=transform,
                                 train=True)
    minist = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=False)
    # 保存MINIST数据为图片
    lable_count = {}#用于给某一类的图片进行编号
    for data in minist:
        img = data[0].clone().numpy()[0][0] # 拷贝数据转换为np数组
        img = img *255 # dataloader中数据以浮点归一化形式存放，需要反归一和改变数据类型
        img = img.astype(np.uint8)
        lable = str(data[1].clone().numpy()[0])
        save_path = './data/minist_data/' + lable
        # 判断save_path是否存在,没有就创建他。
        if not os.path.exists(save_path): #imwrite不能创建文件夹，因此此步骤是必须的。
            os.mkdir(save_path)
        if lable in lable_count:
            if (lable_count[lable] >= TOTAL_NUM):#达到训练集大小要求，不再对这一类生成图片
                continue
            lable_count[lable] = lable_count[lable] + 1
        else:
            lable_count[lable] = 1
        img_path = save_path + '/' + str(lable_count[lable]) + '.bmp'
        # 图像二值化处理
        retval, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)  # 阈值设为127
        cv.imwrite(img_path, binary_img)  # 保存二值化图片
#=======进行仿射变换获得图片以便于传统方法识别======#
def affine_data():
    #仿射变换数据集增强
    repaeat_times = 10
    for i in range(0, 10):
        iTen = glob.glob('data/small_data/' + str(i) + '/*.*')  # 使用glob函数获得满足的所有文件名
        times = 0
        for file in iTen:
            for j in range(repaeat_times):
                times = times + 1
                img = cv.imread(file, 0)
                height, width = img.shape[:2]
                angle = random.randint(-60,60)#倾斜范围-60~60
                scale = 0.5 + random.random()#缩放范围0.5~1.5
                M0 = cv.getRotationMatrix2D((width / 2, height / 2), angle , scale)
                newimg = cv.warpAffine(img, M0, (width, height))
                save_path = 'data/affine_data/' + str(i)
                if not os.path.exists(save_path):  # imwrite不能创建文件夹，因此此步骤是必须的。
                    os.mkdir(save_path)
                img_path = save_path + '/' + str(times)+'.bmp'
                cv.imwrite(img_path,newimg)
    return

#######################################
#==============传统方法================#
######################################

#====SVM方法，构造数据集、设置svm模型、使用svm模型====#
def SVM():
    #----------获取训练数据----------------
    trainData, trainLabels, testData, testLabels = getData()
    # ----------构造svm------------------
    svm = cv.ml.SVM_create()  # 创建一个SVM实例，一般用于线性的二分类问题
    svm.setKernel(cv.ml.SVM_LINEAR)  # 设置kernel类型，可以使用非线性的核函数实现实际上的非线性分类
    svm.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)  # 训练svm
    # ----------使用svm------------------
    result = svm.predict(testData)[1]  # 获取识别标签
    mask =  result == testLabels  # 比较识别结果是否等于实际标签
    correct = np.count_nonzero(mask)  # 计算非零值（相等）的个数
    accuracy = correct * 100.0 / result.size  # 计算准确率（相等个数/全部）
    print("识别准确率为：",accuracy)
    return accuracy

#====getData函数，获取训练数据、测试数据及对应标签（预先倾斜矫正和获取HOG值）====#
def getData():
    data=[]   # 存储所有数字的所有图像
    for i in range(0,10):
        # 使用glob函数获得满足的所有文件名
        if (DATESET_FLAG == 0):
            iTen = glob.glob('data/small_data/'+str(i)+'/*.*')#原本的图片作为数据集
        if (DATESET_FLAG == 1):
            iTen = glob.glob('data/minist_data/'+str(i)+'/*.*')#MINIST中的图片作为数据集
        if (DATESET_FLAG == 2):
            iTen = glob.glob('data/affine_data/' + str(i) + '/*.*')  # 仿射变换增强的图片作为数据集
        num=[]      # 临时列表，每次循环用来存储某一个数字的所有图像特征
        for file in iTen:    # 逐个提取文件名
            # step 1:预处理（读取图像，色彩转换、大小转换）
            image=cv.imread(file,0)   # 逐个读取文件，放入image中
            #image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)   # 彩色——>灰色
            #必要时需要做反色处理：前景背景切换  #x=255-x
            image=cv.resize(image,(20,20))   # 调整大小
            # step2：倾斜校正
            if (DESKEW_FLAG == 1):
                image=deskew(image)   # 倾斜校正
            # step3：获取hog值
            hogValue=hog(image)   # 获取hog值
            num.append(hogValue)  # 把当前图像的hog值放入num中
        data.append(num)  # 把单个数字的所有hogValue放入data，每个数字所有hog值占一行 TOTAL_NUM*10
    x=np.array(data)
    # step4：划分数据集（训练集、测试集）
    trainData=np.float32(x[:,:TRAIN_NUM])#每个数字的0~TRAIN_NUM-1范围内的图片的HOG作为训练集
    testData=np.float32(x[:,TRAIN_NUM:])#每个数字的=TRAIN_NUM~TOTAL_NUM范围内的图片的HOG作为训练集
    # step5：塑形，调整为64列
    trainData=trainData.reshape(-1,64)    #-1表示自动的适配行，按照内存中的顺序（先行后列）。因此每TRAIN_NUM个元素对应一个数字不同图像的HOG值
    testData=testData.reshape(-1,64)     #同理，每10-TRAIN_NUM个元素对应一个数字不同图像的HOG值
    # step6：打标签
    trainLabels = np.repeat(np.arange(10),TRAIN_NUM)[:,np.newaxis]      #训练图像贴标签,重复TRAIN_NUM次（这TRAIN_NUM个是同一个数字）
    testLabels = np.repeat(np.arange(10),TEST_NUM)[:,np.newaxis]       #测试图像贴标签
    return  trainData,trainLabels,testData,testLabels

#====抗扭斜函数，通过一个仿射变换实现对图倾斜度的纠正====#
def deskew(img):
    #怎么选择一个合适的倾斜度度量？ 基于统计的方法。
    m = cv.moments(img)#计算图的x和y两个随机变量在三阶以下的矩，这些数据被用来描述形状、位置等信息
    '''
    m中包含空间矩(e.g. m00)、中心矩(e.g. mu02 , mu11)和归一化中心矩(e.g. nu01)。
    mu02：y坐标的二阶中心矩，描述了在y轴上的倾斜情况。 mu11：x和y坐标的混合二阶中心矩。
    倾斜度是mu11/mu02
    '''
    if abs(m['mu02']) < 1e-2:
        return img.copy()#在y上的二阶中心矩很小，说明倾斜可以忽略不计
    skew = m['mu11']/m['mu02']#计算偏斜度
    height, width = img.shape[:2]
    size=(height,width)   #每个数字的图像的尺寸
    s=width
    #怎么由倾斜度决定仿射矩阵的取值？
    M = np.float32([[1, skew, -0.5*s*skew], [0, 1, 0]])#设置仿射矩阵，用于恢复倾斜
    affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
    img = cv.warpAffine(img,M,size,flags=affine_flags)#应用仿射矩阵
    return img

#====HOG函数，计算图的方向梯度直方图====#
def hog(img):
    #使用sobel算子计算水平和垂直梯度
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)#梯度转化为极坐标形式
    bins = np.int32(16*ang/(2*np.pi))#规约弧度到16个刻度之间
    #分区计算方向和幅值的直方图统计
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(),16) for b, m in zip(bin_cells, mag_cells)]
    #zip捆绑迭代，ravel展平数组，bincount计算方向值0-15出现的频率然后乘以该点的幅度
    hist = np.hstack(hists)#hstack链接为（16+16+16+16） = 64
    return hist

#######################################
#============机器学习方法===============#
######################################

#对张量的图像数据进行倾斜校正
def deskew_tensor(tensor):
    img = np.array(tensor[0])
    img = deskew(img)
    new_tensor = torch.tensor(img)
    new_tensor = new_tensor.reshape((1, 28, 28))
    return new_tensor

def Torch():
    # 1. 数据准备
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    if DESKEW_FLAG == 1:#开启倾斜矫正
        transform=transforms2.Compose([transforms2.ToTensor(),
                                       transforms2.Lambda(deskew_tensor),
                                       transforms2.Normalize(mean=[0.5],std=[0.5])])
    if DESKEW_FLAG == 0:#关闭倾斜矫正
        transform = transforms2.Compose([transforms2.ToTensor(),
                                         transforms2.Normalize(mean=[0.5], std=[0.5])])
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
    EPOCH=10
    Train_Accuracy = []
    Test_Accuracy = []
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
        Train_Accuracy.append(train_Accuracy.item())
        Test_Accuracy.append(test_Accuracy.item())
        print("Epoch {:d}/{:d}: Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(t+1,EPOCH,loss,train_Accuracy,test_Accuracy))
    return Train_Accuracy, Test_Accuracy

if __name__ == "__main__":
    main()