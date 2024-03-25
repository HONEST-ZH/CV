import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

def main():
    #读取图像，处理图像
    img = cv.imread('cat.png',1)#以彩色格式打开图片
    R,G,B =cv.split(img)#分割色彩空间，对分量做高斯模糊
    # 计算n*n的方差为d的高斯滤波卷积核
    kernel = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            val = gauss(i - mid, mid - j, d)
            kernel[i][j] = val
    nm_kernel = normalizaton(kernel) #对卷积核进行归一化
    #对三个通道分别进行高斯模糊
    newR = conv(R, nm_kernel)
    newG = conv(G, nm_kernel)
    newB = conv(B, nm_kernel)
    #合并图片
    new_img = cv.merge([newR,newG,newB])
    gauss_img = cv.GaussianBlur(img,(n,n),d) #使用cv工具高斯滤波
    #生成对比图
    fig, axs = plt.subplots(1, 3,figsize=(16,8))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.suptitle("GaussianBlur Result")
    axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), aspect = 'equal', interpolation = 'none')
    axs[0].axis('off')
    axs[0].set_title("Original picture")
    axs[1].imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB), aspect = 'equal', interpolation = 'none')
    axs[1].axis('off')
    axs[1].set_title("Result of my gauss\n (3*3,d=1.5)")
    axs[2].imshow(cv.cvtColor(gauss_img, cv.COLOR_BGR2RGB), aspect = 'equal', interpolation = 'none')
    axs[2].axis('off')
    axs[2].set_title("Result of cv.GaussianBlur\n (3*3,d=1.5)")
    fig.savefig("Result.png")
    # 对比不同的卷积核大小
    gauss_img1 = cv.GaussianBlur(img, (5, 5), 1.5)  # 使用cv工具高斯滤波
    gauss_img2 = cv.GaussianBlur(img, (7, 7), 1.5)  # 使用cv工具高斯滤波
    gauss_img3 = cv.GaussianBlur(img, (9, 9), 1.5)  # 使用cv工具高斯滤波
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.suptitle("GaussianBlur at different kernel size")
    axs[0, 0].imshow(cv.cvtColor(gauss_img, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Result of cv.GaussianBlur\n (3*3,d=1.5)")
    axs[0, 1].imshow(cv.cvtColor(gauss_img1, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Result of cv.GaussianBlur\n (5*5,d=1.5)")
    axs[1, 0].imshow(cv.cvtColor(gauss_img2, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("Result of cv.GaussianBlur\n (7*7,d=1.5)")
    axs[1, 1].imshow(cv.cvtColor(gauss_img3, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Result of cv.GaussianBlur\n (9*9,d=1.5)")
    fig.savefig("Kernel_size.png")
    # 对比不同的方差
    gauss_img4 = cv.GaussianBlur(img, (9, 9), 3)  # 使用cv工具高斯滤波
    gauss_img5 = cv.GaussianBlur(img, (9, 9), 6)  # 使用cv工具高斯滤波
    gauss_img6 = cv.GaussianBlur(img, (9, 9), 12)  # 使用cv工具高斯滤波
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.suptitle("GaussianBlur at different kernel size")
    axs[0, 0].imshow(cv.cvtColor(gauss_img3, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Result of cv.GaussianBlur\n (9*9,d=1.5)")
    axs[0, 1].imshow(cv.cvtColor(gauss_img4, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Result of cv.GaussianBlur\n (9*9,d=3)")
    axs[1, 0].imshow(cv.cvtColor(gauss_img5, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("Result of cv.GaussianBlur\n (9*9,d=6)")
    axs[1, 1].imshow(cv.cvtColor(gauss_img6, cv.COLOR_BGR2RGB), aspect='equal', interpolation='none')
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Result of cv.GaussianBlur\n (9*9,d=12)")
    fig.savefig("Var.png")
def gauss(x,y,d):
    a = math.exp(-(x*x+y*y)/(2*d*d))#计算分子
    b = 1/(2*math.pi*d*d)#计算分母
    res = a*b
    return res
def normalizaton(kernel):
    #question:是归一化为平方和为1还是和为1？后者
    nm_kernel = kernel / np.sum(kernel)
    return nm_kernel;
def conv(img,nm_kernel):
    # 对img和归一化后的高斯卷积核nm_kernel进行卷积，存到new_img
    new_img = np.zeros_like(img)  # 创建滤波后的图像
    for i in range(mid, img.shape[0] - mid):
        for j in range(mid, img.shape[1] - mid):
            sum = 0
            for s in range(-mid, mid + 1):
                for t in range(-mid, mid + 1):
                    sum = sum + img[i + s][j + t] * nm_kernel[mid + s][mid + t]
            new_img[i][j] = sum
    return new_img
if __name__ == "__main__":
    #设计高斯滤波的卷积核，大小为n*n,方差为d.
    n = 3
    mid = int((n - 1) / 2)
    d = 1.5
    main()