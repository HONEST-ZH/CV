import os
import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from mtcnn_pytorch.src import visualization_utils
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":

    # 图片路径
    picturePath = 'picture.jpeg'

    # 打开图片
    img = cv2.imread(picturePath)
    # cv2.imshow("picture",img)
    # cv2.waitKey()

    # 模型声明
    mtcnn = MTCNN()  # ValueError: Object arrays cannot be loaded when allow_pickle=False ->pip install numpy==1.16.2

    # 人脸检测
    try:
        p = Image.fromarray(img[..., ::-1])  # rgb ->bgr
        boxes, faces,landmarks = mtcnn.align_multi(p)  # bgr -> rgb

        handle_img = visualization_utils.draw_boxes_and_landmarks_on_image(p,boxes,landmarks)  # 对人脸画框
        draw_img = np.array(handle_img)[...,::-1]  # rgb->bgr
        cv2.imshow(" ",draw_img)
        cv2.waitKey(0)

        # 画框图片保存
        cv2.imwrite("result\\drawPicture.jpg",draw_img)
        print("结果保存于result文件")
    except:
        # 没有检测到人脸视频帧保存
        print('no face captured')


