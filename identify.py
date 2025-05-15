import cv2
import numpy as np


def is_circle(contour, circularity_thresh=0.8):
    """
    判断轮廓是否为近似圆形，返回 1 表示是圆形，0 表示不是。

    参数:
        contour: OpenCV轮廓
        circularity_thresh: 圆形度阈值，默认0.8

    返回:
        1：为圆形
        0：不为圆形
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    circularity = 4 * np.pi * area / (perimeter ** 2)

    return int(circularity >= circularity_thresh)

step = 1

cap = cv2.VideoCapture(0)

# 设置卷积核大小
kernel = np.ones((5, 5), np.uint8)

# 红1阈值
up_red1 = np.array([10, 255, 255], dtype=np.uint8)
low_red1 = np.array([0, 43, 46], dtype=np.uint8)

# 红2阈值
up_red2 = np.array([180, 255, 255], dtype=np.uint8)
low_red2 = np.array([156, 43, 46], dtype=np.uint8)

max_red = 0
index_red = -1

# 蓝阈值
up_blue = np.array([124, 255, 255], dtype=np.uint8)
low_blue = np.array([100, 43, 46], dtype=np.uint8)

max_blue = 0
index_blue = -1

max_blue_circle = 0
index_blue_circle = -1

# 创建二维码检测器
qr_detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, 1)
    if not ret:
        break


    output_image = image.copy()
    
    # 中值滤波
    median = cv2.medianBlur(image, 3)

    # BGR转HSV
    HSV = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    if step == 1:

        # 设置掩膜
        mask_red1 = cv2.inRange(HSV, low_red1, up_red1)
        mask_red2 = cv2.inRange(HSV, low_red2, up_red2)

        # 合并掩膜
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 开闭运算
        open_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        close_red = cv2.morphologyEx(open_red, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours_red, _ = cv2.findContours(close_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_red is not None:
            # 将轮廓保存为数组
            contours_array_red = [cnt_red for cnt_red in contours_red]  # 将轮廓点集存储为一个数组

            # 遍历轮廓并计算面积
            for i in range(len(contours_red)):
                area_red = cv2.contourArea(contours_array_red[i])
                if area_red > max_red:
                    max_red = area_red
                    index_red = i
            
            if index_red != -1:

                # 找到最大红色色块轮廓
                res_red = cv2.drawContours(output_image, contours_array_red[index_red], -1, [0, 255, 0], 4)
                
                # 计算最大红色图形轮廓的中心点
                (x, y), r = cv2.minEnclosingCircle(contours_array_red[index_red])
                center_red = (int(x), int(y))

                print("最大红色色块：{}".format(center_red))
                cv2.circle(output_image, center_red, 3, (0, 255, 0), 5)
                
                max_red = 0
                index_red = -1
                step = 2

            else:
                print("未检测到红色色块")

        else:
            print("未检测到红色色块")
        


    elif step == 2:

        # 设置掩膜
        mask_blue = cv2.inRange(HSV, low_blue, up_blue)

        # 开闭运算
        open_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        close_blue = cv2.morphologyEx(open_blue, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours_blue, _ = cv2.findContours(close_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_blue is not None:
            # 将轮廓保存为数组
            contours_array_blue = [cnt_blue for cnt_blue in contours_blue]  # 将轮廓点集存储为一个数组

            # 遍历轮廓并计算面积
            for j in range(len(contours_blue)):
                area_blue = cv2.contourArea(contours_array_blue[j])
                if area_blue > max_blue:
                    max_blue = area_blue
                    index_blue = j
            
            if index_blue != -1:

                # 找到轮廓
                res_blue = cv2.drawContours(output_image, contours_array_blue[index_blue], -1, [0, 255, 0], 4)
                
                # 计算最大蓝色图形轮廓的中心点
                (x, y), r = cv2.minEnclosingCircle(contours_array_blue[index_blue])
                center_blue = (int(x), int(y))

                print("最大蓝色色块：{}".format(center_blue))
                cv2.circle(output_image, center_blue, 3, (0, 255, 0), 5)
                
                max_blue = 0
                index_blue = -1
                step = 3

            else:
                print("未检测到蓝色色块")
        
        else:
            print("未检测到蓝色色块")


    elif step == 3:

        # 设置掩膜
        mask_blue = cv2.inRange(HSV, low_blue, up_blue)

        # 开闭运算
        open_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        close_blue = cv2.morphologyEx(open_blue, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours_blue, _ = cv2.findContours(close_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_blue_circle = []

        # 遍历轮廓并判断是否为圆形
        for k in range(len(contours_blue)):
            if is_circle(contours_blue[k]):
                contours_blue_circle.append(contours_blue[k])
        
        if len(contours_blue_circle) != 0:

            # 遍历轮廓并计算面积
            for l in range(len(contours_blue_circle)):
                area_blue_circle = cv2.contourArea(contours_blue_circle[l])
                if area_blue_circle > max_blue_circle:
                    max_blue_circle = area_blue_circle
                    index_blue_circle = l
            
            if index_blue_circle != -1:

                # 找到轮廓
                res_blue_circle = cv2.drawContours(output_image, contours_blue_circle[index_blue_circle], -1, [0, 255, 0], 4)

                # 计算最大蓝色圆形轮廓的中心点
                (x, y), r = cv2.minEnclosingCircle(contours_blue_circle[index_blue_circle])
                center_blue_circle = (int(x), int(y))

                print("最大蓝色圆圈：{}".format(center_blue_circle))
                cv2.circle(output_image, center_blue_circle, 3, (0, 255, 0), 5)

                max_blue_circle = 0
                index_blue_circle = -1
                step = 4

            else:
                print("未检测到蓝色圆圈")

        else:
            print("未检测到蓝色圆圈")


    elif step == 4:

        # 解码二维码
        data, points, _ = qr_detector.detectAndDecode(image)
        
        # 条件：必须同时检测到二维码位置且识别出内容
        if points is not None and data and data.startswith("http"):
            # 转换坐标并画出二维码轮廓
            points = np.int32(points).reshape(-1, 2)
            cv2.polylines(output_image, [points], isClosed=True, color=(0, 255, 0), thickness=3)

            print("二维码:({})".format(data))

            # 计算二维码中心坐标
            center_x = int(np.mean(points[:, 0]))  # 计算x坐标的均值
            center_y = int(np.mean(points[:, 1]))  # 计算y坐标的均值
            
            print("二维码中心坐标: ({}, {})".format(center_x, center_y))

            # 绘制二维码中心点
            cv2.circle(output_image, (center_x, center_y), 5, (0, 255, 0), -1)
            
            step = 1
            
        else:
            print("未检测到二维码或解码失败")
    

    # 显示窗口
    cv2.imshow("output_image", output_image)

    # 按下 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
