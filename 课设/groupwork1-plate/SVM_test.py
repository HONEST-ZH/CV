import random
import cv2
import numpy as np
import glob

#----------STEP4.1 建立模板，部分省份，使用字典表示---------
templateDict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
            10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',
            18:'J',19:'K',20:'L',21:'M',22:'N',23:'P',24:'Q',25:'R',
            26:'S',27:'T',28:'U',29:'V',30:'W',31:'X',32:'Y',33:'Z',
            34:'京',35:'津',36:'冀',37:'晋',38:'蒙',39:'辽',40:'吉',41:'黑',
            42:'沪',43:'苏',44:'浙',45:'皖',46:'闽',47:'赣',48:'鲁',49:'豫',
            50:'鄂',51:'湘',52:'粤',53:'桂',54:'琼',55:'渝',56:'川',57:'贵',
            58:'云',59:'藏',60:'陕',61:'甘',62:'青',63:'宁',64:'新'}
            #65:,'港',66:'澳',67:'台'}
#-----------STEP4.2 获取所有字符的路径信息------------
def getcharacters():
    c=[]
    for i in range(0,65):
        words=[]
        words.extend(glob.glob('template/'+templateDict.get(i)+'/*.*'))
        #words.extend(glob.glob('data/' + templateDict.get(i) + '/*.*'))
        c.append(words)
    return c
####方法二（支持向量机）####
def svm(chars):
    # ----------获取训练数据----------------
    trainData, trainLabels, testData, testLabels = getData(chars)
    # ----------构造svm------------------
    svm = cv2.ml.SVM_create()  # 创建一个SVM实例，一般用于线性的二分类问题
    svm.setKernel(cv2.ml.SVM_LINEAR)  # 设置kernel类型，可以使用非线性的核函数实现实际上的非线性分类
    svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)  # 训练svm
    # ----------使用svm------------------
    #分训练集和测试集时
    result = svm.predict(testData)[1]  # 获取识别标签
    mask = result == testLabels  # 比较识别结果是否等于实际标签
    correct = np.count_nonzero(mask)  # 计算非零值（相等）的个数
    accuracy = correct * 100.0 / result.size  # 计算准确率（相等个数/全部）
    print("识别准确率为：", accuracy)
    return accuracy
def getData(chars):
    data = []  # 存储所有数字的所有图像
    size = 20# 数据集不均匀（字符模板集中模板数量不同！） 添加或删除一些数据，统一数据的规模
    for char in chars:
        num = []  # 临时列表，每次循环用来存储某一个字符的所有图像特征
        length = len(char)
        if length > size:
            char = char[:size]
        else:
            # 直接复制，可能导致学习特征发生偏移，偏向某类中的一个样本
            while length != size:
                char.append(char[random.randint(0, length-1)])
                length = length + 1
        for word in char:  # 逐个提取文件名
            image = cv2.imdecode(np.fromfile(word, dtype=np.uint8), 0)
            hogValue = hog(image)  # 获取hog值，64列的数组
            num.append(hogValue)  # 把当前图像的hog值放入num中，size*64
        data.append(num)  # 把单个字符的所有hogValue放入data,68*size*64
    x = np.array(data)
    #分割训练和测试时
    trainData = np.float32(x[:, 5:])
    testData = np.float32(x[:, :5])
    trainData = trainData.reshape(-1,64)  # 每size个元素对应一个字符不同图像的HOG值
    testData = testData.reshape(-1,64)
    trainLabels = np.repeat(np.arange(65),15)[:,np.newaxis]    #训练图像贴标签
    testLabels = np.repeat(np.arange(65), 5)[:, np.newaxis]
    return trainData, trainLabels, testData, testLabels

def hog(img):
    # 使用sobel算子计算水平和垂直梯度
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)  # 梯度转化为极坐标形式
    bins = np.int32(16 * ang / (2 * np.pi))  # 规约弧度到16个刻度之间
    # 分区计算方向和幅值的直方图统计
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
    # zip捆绑迭代，ravel展平数组，bincount计算方向值0-15出现的频率然后乘以该点的幅度
    hist = np.hstack(hists)  # hstack链接为（16+16+16+16） = 64
    return hist

# =========主程序=============
if __name__ == "__main__":
    chars=getcharacters()
    accuracy = svm(chars)