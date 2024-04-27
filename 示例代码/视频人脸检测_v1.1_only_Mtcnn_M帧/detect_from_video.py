import os
import cv2
import argparse
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from pathlib import Path
from mtcnn_pytorch.src import visualization_utils
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
    # =================================================参数设定===============================================================

    # 解析器
    parser = argparse.ArgumentParser(description='take a picture from video')
    parser.add_argument('--video_name', '-i', default='IPartment1.mp4', type=str, help='the dataset source')
    parser.add_argument('--frameInterval','-m',default=60,type=int,help='frame interval')
    args = parser.parse_args()

    # ======================================================================================================================
    # ============================================视频路径与图片保存路径=========================================================

    # 绝对路径
    ROOT_PATH = os.path.join(os.path.abspath(__file__), "..")  # this is a file path

    # 视频路径
    video_data_path = Path(os.path.join(ROOT_PATH, "video", args.video_name))
    if not video_data_path.exists():
        raise FileNotFoundError("{}没有放在./video/目录下".format(args.video_name))

    # 保存图片路径
    save_path = os.path.join(ROOT_PATH, "video_output")
    save_path = Path(save_path)  # 路径标准化

    # 判断路径是否存在，存在说明以及检测过，删去
    save_path = save_path / args.video_name.split(".")[0]
    if not save_path.exists():
        # 创建目录
        os.makedirs(save_path, exist_ok=True)
    else:
        print("已经检测过视频")
        del_list = os.listdir(str(save_path))
        for f in del_list:
            file_path = os.path.join(str(save_path), f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # ======================================================================================================================
    # ============================================视频对象与其信息=============================================================
    # 获取video
    video = cv2.VideoCapture(str(video_data_path))

    # 视频帧率
    fps = video.get(cv2.CAP_PROP_FPS)

    # 视频总帧数
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # 视频宽度,视频高度
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 打印视频信息
    print("视频信息如下:")
    print("帧率:{0}\n帧数:{1}\n帧像素:{2}x{3}\n".format(fps,frameCount,size[0],size[1]))

    # 读取第一帧
    success, frame = video.read()

    # ======================================================================================================================
    # ============================================模型声明====================================================================

    # 模型声明
    mtcnn = MTCNN()  # ValueError: Object arrays cannot be loaded when allow_pickle=False ->pip install numpy==1.16.2

    # ======================================================================================================================
    # ============================================视频帧人脸检测===============================================================

    i = 0
    # 对每帧进行人脸检测
    while success:
        if i%args.frameInterval== 0:
            try:
                p = Image.fromarray(frame[..., ::-1])  # rgb ->bgr
                boxes, faces,landmarks = mtcnn.align_multi(p)  # bgr -> rgb

                handle_frame = visualization_utils.draw_boxes_and_landmarks_on_image(p,boxes,landmarks)  # 对视频帧人脸画框
                draw_frame = np.array(handle_frame)[...,::-1]  # rgb->bgr
                # cv2.imshow(" ",draw_frame)
                # cv2.waitKey(0)

                # 画框视频帧保存
                filename = str(save_path / '{}.jpg'.format(i))
                print("{} frame:face captured保存第{}帧".format(i,i))
                cv2.imencode('.jpg', draw_frame)[1].tofile(filename)
            except:
                # 没有检测到人脸视频帧保存
                filename = str(save_path / '{}.jpg'.format(i))
                print('{} frame:no face captured'"保存第{}帧".format(i,i))
                cv2.imencode('.jpg', frame)[1].tofile(filename)
        i += 1
        success, frame = video.read()

    print("finish!")
    # 释放视频资源
    video.release()
