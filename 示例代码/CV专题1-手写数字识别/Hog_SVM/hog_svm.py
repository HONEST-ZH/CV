#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import glob


# In[8]:


#=============抗扭斜函数=================
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    s=20
    M = np.float32([[1, skew, -0.5*s*skew], [0, 1, 0]])
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    size=(20,20)   #每个数字的图像的尺寸
    img = cv2.warpAffine(img,M,size,flags=affine_flags)
    return img


# In[9]:


#=============HOG函数=================
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(16*ang/(2*np.pi))    
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(),16) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) 
    return hist


# In[10]:


TRAIN_NUM=1
TEST_NUM=9
#=============getData函数，获取训练数据、测试数据及对应标签=================
def getData():
    data=[]   #存储所有数字的所有图像
    for i in range(0,10):
        iTen=glob.glob('data/'+str(i)+'/*.*')   # 所有图像的文件名
        num=[]      #临时列表，每次循环用来存储某一个数字的所有图像
        for number in iTen:    #逐个提取文件名
            # step 1:预处理（读取图像，色彩转换、大小转换）
            image=cv2.imread(number)   #逐个读取文件，放入image中
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   #彩色——>灰色
            # x=255-x   #必要时需要做反色处理：前景背景切换
            image=cv2.resize(image,(20,20))   #调整大小
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


# In[11]:


#=============SVM函数，构造svm模型、使用svm模型=================
def SVM(trainData,trainLabels,testData,TestLabels):
    #----------构造svm------------------
    svm = cv2.ml.SVM_create()               # 初始化
    svm.setKernel(cv2.ml.SVM_LINEAR)        # 设置kernel类型
    svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)  #训练svm
    #----------使用svm------------------
    result = svm.predict(testData)[1]       #获取识别标签
    mask = result==TestLabels               #比较识别结果是否等于实际标签
    correct = np.count_nonzero(mask)        #计算非零值（相等）的个数
    accuracy = correct*100.0/result.size    #计算准确率（相等个数/全部）
    return accuracy


# In[12]:


#=============主程序=================
trainData,trainLabels,testData,TestLabels=getData()
accuracy=SVM(trainData,trainLabels,testData,TestLabels)
print("识别准确率为：",accuracy)

