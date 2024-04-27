import cv2
import os
from pathlib import Path
import argparse

# 解析器
parser = argparse.ArgumentParser()
parser.add_argument('--video_frame', '-i', default='IPartment1', type=str, help='frames of video')
args = parser.parse_args()

# 视频帧路径与视频保存路径
image_folder = os.path.join('video_output',args.video_frame)
video_save_path = 'generated_video'

# 绝对路径
ROOT_PATH = os.path.join(os.path.abspath(__file__), "..")

# 视频路径
save_path = Path(os.path.join(ROOT_PATH, video_save_path))

if not save_path.exists():
    # 创建目录
    os.makedirs(save_path, exist_ok=True)

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")], key=lambda x: int(x.split(".")[0]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 使用VideoWriter对象创建视频
video_name = image_folder.split("\\")[1]+".mp4"
video = cv2.VideoWriter(os.path.join(save_path,video_name), 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

print("视频生成完成！")
# 释放VideoWriter对象并销毁所有窗口
cv2.destroyAllWindows()
video.release()