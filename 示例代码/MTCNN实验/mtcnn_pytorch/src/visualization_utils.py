import numpy as np
import cv2

def draw_boxes_and_landmarks_on_image(image, boxes, landmarks):
    if len(boxes)!=0 and len(landmarks)!=0:
        image = np.array(image)
        # plt.imshow(image)
        # plt.show()
        for box, landmark in zip(boxes, landmarks):
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            score = round(score, 2)

            # 在图像上绘制方框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 显示概率大小
            text = f'Probability: {score}'
            text_position = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 255, 0)
            thickness = 1
            cv2.putText(image, text, text_position, font, font_scale, color, thickness)

            # 绘制特征点
            landmark_reshaped = landmark.reshape(5, 2)
            for point in landmark_reshaped:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 1, (0, 0, 255), 1)


        return image
