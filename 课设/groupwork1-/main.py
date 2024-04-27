import cv2
# from matplotlib import pyplot as plt
import numpy as np
import glob

#========STEP1.预处理：图像去噪等处理==========#

def preprocessor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 色彩空间转换（RGB-->GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)# 去噪处理
    return image

#=======STEP2.车牌定位=======#

def getPlate(image):
    rawImage=image.copy()
    image = preprocessor(image)
    # Sobel算子（X方向边缘梯度）
    Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)  # 映射到[0.255]内
    image = absX
    # 阈值处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # 闭运算：先膨胀后腐蚀，车牌各个字符是分散的，让车牌构成一体
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
    # 开运算：先腐蚀后膨胀，去除噪声
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelY)
    # 中值滤波：去除噪声
    image = cv2.medianBlur(image, 15)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #测试语句，查看处理结果
    # image = cv2.drawContours(rawImage.copy(), contours, -1, (0, 0, 255), 3)
    # cv2.imshow('imagecc', image)
    #逐个遍历，将宽度>3倍高度的轮廓确定为车牌
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if weight > (height * 3):
            plate = rawImage[y:y + height, x:x + weight]
    return plate

#=======STEP3.车牌分割===========#

# ------STEP3.1 让一个字构成一个整体---------
def GetOne(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 阈值处理（二值化）
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    # 膨胀处理，让一个字构成一个整体（大多数字不是一体的，是分散的）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    return image
# -----STEP3.2 拆分车牌函数，将车牌内各个字符分离-----
def splitPlate(image):
    # 查找轮廓，各个字符的轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    # 遍历所有轮廓
    for item in contours:
        rect = cv2.boundingRect(item)
        words.append(rect)
    # print(len(contours))  #测试语句：看看找到多少个轮廓
    #-----测试语句：看看轮廓效果-----
    # imageColor=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # x = cv2.drawContours(imageColor, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("contours",x)
    #-----测试语句：看看轮廓效果-----
    # 按照x轴坐标值排序（自左向右排序）
    words = sorted(words,key=lambda s:s[0],reverse=False)
    # 用word存放左上角起始点及长宽值
    plateChars = []
    for word in words:
        # 筛选字符的轮廓(高宽比在1.5-8之间，宽度大于3)
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 8)) and (word[2] > 3):
            plateChar = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            plateChars.append(plateChar)
    # 测试语句：查看各个字符
    # for i,im in enumerate(plateChars):
    #     cv2.imshow("char"+str(i),im)
    return plateChars

#=======STEP4.字符识别===========#

######方法一（最佳匹配）#########

#----------STEP4.1模板，部分省份，使用字典表示---------
templateDict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
            10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',
            18:'J',19:'K',20:'L',21:'M',22:'N',23:'P',24:'Q',25:'R',
            26:'S',27:'T',28:'U',29:'V',30:'W',31:'X',32:'Y',33:'Z',
            34:'京',35:'津',36:'冀',37:'晋',38:'蒙',39:'辽',40:'吉',41:'黑',
            42:'沪',43:'苏',44:'浙',45:'皖',46:'闽',47:'赣',48:'鲁',49:'豫',
            50:'鄂',51:'湘',52:'粤',53:'桂',54:'琼',55:'渝',56:'川',57:'贵',
            58:'云',59:'藏',60:'陕',61:'甘',62:'青',63:'宁',64:'新',
            65:'港',66:'澳',67:'台'}
#-----------STEP获取所有字符的路径信息------------
def getcharacters():
    c=[]
    for i in range(0,67):
        words=[]
        words.extend(glob.glob('template/'+templateDict.get(i)+'/*.*'))
        c.append(words)
    return c
#----------STEP计算匹配值函数-------------
def getMatchValue(template,image):
    #读取模板图像
    # templateImage=cv2.imread(template)   #cv2读取中文文件名不友好
    templateImage=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
    #模板图像色彩空间转换，BGR-->灰度
    templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)
    #模板图像阈值处理， 灰度-->二值
    ret, templateImage = cv2.threshold(templateImage, 0, 255, cv2.THRESH_OTSU)
    # 获取待识别图像的尺寸
    height, width = image.shape
    # 将模板图像调整为与待识别图像尺寸一致
    templateImage = cv2.resize(templateImage, (width, height))
    #计算模板图像、待识别图像的模板匹配值
    result = cv2.matchTemplate(image, templateImage, cv2.TM_CCOEFF)
    # 将计算结果返回
    return result[0][0]
#----------STEP对车牌内字符进行识别------------
def matchChars(plates,chars):
    results=[]   #存储所有的识别结果
    #最外层循环：逐个遍历要识别的字符。
    for plateChar in plates:
        bestMatch = []      #要识别的字符最匹配的模板集
        #中间层循环：针对模板集，进行逐个遍历，
        for words in chars:
            match = []      #match，存储的是要识别的字符和对应的模板集中的所有模板的匹配值
            #最内层循环：针对模板集中的的所有模板计算匹配值，找到一个模板集最佳的模板
            for word in words:
                result = getMatchValue(word,plateChar)
                match.append(result)
            bestMatch.append(max(match))   #将模板集中匹配的最好的模板加入bestMatch，代表对应的模板集
        i = bestMatch.index(max(bestMatch))  #i是最佳匹配的模板集的索引值，和chars中的索引值一致
        r = templateDict[i]    #r是单个待识别字符的识别结果
        results.append(r)   #将每一个分割字符的识别结果加入到results内
    return results   #返回所有的识别结果

# =========主程序=============
if __name__ == "__main__":
    #1.初始化
    image = cv2.imread("gua.jpg")                #读取原始图像
    cv2.imshow("original",image)                 #显示原始图像
    #2.车牌定位（长宽比）
    image=getPlate(image)                   #获取车牌
    cv2.imshow('plate', image)              #测试语句：看看车牌定位情况
    #3.车牌分割
    #3.1获得一个字符
    image= GetOne(image)                         #一个字一个整体
    cv2.imshow("GetOne",image)            #测试语句，看看预处理结果
    #3.2车牌分割
    plateChars=splitPlate(image)            #分割车牌，将每个字符独立出来
    for i,im in enumerate(plateChars):      #逐个遍历字符
        cv2.imshow("plateChars"+str(i),im)  #显示分割的字符
    #4.车牌识别（匹配值）
    chars=getcharacters()                   #获取所有模板文件（文件名）
    results=matchChars(plateChars, chars)   #使用模板chars逐个识别字符集plates
    results="".join(results)                #将列表转换为字符串
    print("识别结果为：",results)             #输出识别结果
    cv2.waitKey(0)
    cv2.destroyAllWindows()