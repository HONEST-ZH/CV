import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def main():
    #读图片
    img = cv.imread("space.png", 0)
    hist0 = cv.calcHist([img], [0], None, [256], [0, 256])

    # HE
    he_img = cv.equalizeHist(img)
    hist1 = cv.calcHist([he_img], [0], None, [256], [0, 256])

    # CLAHE
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)
    hist2 = cv.calcHist([clahe_img], [0], None, [256], [0, 256])

    # 伽马值矫正
    gamma = 0.5# 伽马值=0.5
    gamma_img = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    hist3 = cv.calcHist([gamma_img], [0], None, [256], [0, 256])
    gamma = 1# 伽马值=1
    gamma_img1 = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    gamma = 1.5# 伽马值=1.5
    gamma_img2 = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    gamma = 2# 伽马值=2
    gamma_img3 = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    #绘制不同处理方式的对比图
    fig0, axs = plt.subplots(2, 2,figsize=(9,6))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig0.suptitle("Picture after different porcessing")
    axs[0, 0].imshow(img,cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Original picture")
    axs[0, 1].imshow(he_img,cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("HE")
    axs[1, 0].imshow(clahe_img,cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("CLAHE")
    axs[1, 1].imshow(gamma_img,cmap='gray')
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Gamma = 0.5")
    fig0.savefig("Picture.png")

    #绘制直方图
    fig1,axs = plt.subplots(2,2,figsize=(9,6))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig1.suptitle("Histogram after different porcessing")
    axs[0, 0].plot(hist0)
    axs[0, 0].set_title("Original picture")
    axs[0, 1].plot(hist1)
    axs[0, 1].set_title("HE")
    axs[1, 0].plot(hist2)
    axs[1, 0].set_title("CLAHE")
    axs[1, 1].plot(hist3)
    axs[1, 1].set_title("Gamma = 0.5")
    fig1.savefig("Histogram.png")

    # 绘制不同Gamma的对比图
    fig0, axs = plt.subplots(2, 2, figsize=(9, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig0.suptitle("Picture after different Gamma")
    axs[0, 0].imshow(gamma_img, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Gamma = 0.5")
    axs[0, 1].imshow(gamma_img1, cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Gamma = 1")
    axs[1, 0].imshow(gamma_img2, cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("Gamma = 1.5")
    axs[1, 1].imshow(gamma_img3, cmap='gray')
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Gamma = 2")
    fig0.savefig("Gamma.png")

    # 对比CLAHE和Gamma = 0.5
    fig0, axs = plt.subplots(2, 3, figsize=(9, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig0.suptitle("Picture after Gamma/CLAHE")
    #space.png
    axs[0, 0].imshow(gamma_img, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Gamma = 0.5")
    axs[1, 0].imshow(clahe_img, cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title("CLAHE")
    #cat.png
    img1=cv.imread("cat.png",0)
    gamma = 0.5  # 伽马值=0.5
    gamma_img4 = np.array(255 * (img1 / 255) ** gamma, dtype='uint8')
    clahe_img1 = clahe.apply(img1)
    axs[0, 1].imshow(gamma_img4, cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Gamma = 0.5")
    axs[1, 1].imshow(clahe_img1, cmap='gray')
    axs[1, 1].axis('off')
    axs[1, 1].set_title("CLAHE")
    #city.jpg
    img2 = cv.imread("city.jpg", 0)
    gamma = 0.5  # 伽马值=0.5
    gamma_img5 = np.array(255 * (img2 / 255) ** gamma, dtype='uint8')
    clahe_img2 = clahe.apply(img2)
    axs[0, 2].imshow(gamma_img5, cmap='gray')
    axs[0, 2].axis('off')
    axs[0, 2].set_title("Gamma = 0.5")
    axs[1, 2].imshow(clahe_img2, cmap='gray')
    axs[1, 2].axis('off')
    axs[1, 2].set_title("CLAHE")
    fig0.savefig("Gamma_CLAHE.png")
    # plt.show()

if __name__ == "__main__":
    main()