import cv2
import os
import argparse
import time

# 解析器
parser = argparse.ArgumentParser(description='play video')
parser.add_argument('--video', '-i', default='IPartment1.mp4', type=str, help='face_video')
args = parser.parse_args()

# 打开视频文件
video_file = os.path.join('generated_video',args.video)
cap = cv2.VideoCapture(video_file)

# 检查视频文件是否成功打开
if not cap.isOpened():
    print("Error opening video file")
    exit()

# 播放视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    time.sleep(1)

    # 如果按下 'q' 键，则退出循环
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
