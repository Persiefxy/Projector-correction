import sys,os  # Import sys module for path manipulation
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # Add parent directory to system
import method.camera_screen as cs
import cv2
import numpy as np
rate=0.08
# 读取图像
cam= cs.Camera("uvc")
img =  cam.read()


# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("【2】灰度图", gray)

# Canny 边缘检测
edges = cv2.Canny(gray, 10, 70)

# 找到轮廓
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

img2 = img.copy()
# 遍历所有轮廓
for cnt in contours:
    cv2.drawContours(img2, [cnt], 0, (0, 0, 255), 2)
# 保存结果
cv2.imshow("All Contours", img2)
cv2.imshow("All edges", edges)
height, width = edges.shape[:2]
# 对轮廓图进行霍夫直线检测
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=250)


accu = min(width*rate, height*rate)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, accu, accu, 0)
"""其主要参数含义如下:
edges: 输入的边缘图像,一般是通过Canny等边缘检测算子得到。
1: 距离分辨率。表示线段上点到直线的距离精度。
np.pi/180: 角度分辨率。表示线段的角度精度。这里设置为1度。
30: 阈值,用于判断一条线段至少需要多少像素点投票才能确定是一条线段。
accu: 最小线段长度。过滤出大于此长度的线段。
0: 最大间隙。允许线段上的点之间的最大间隔,可以填补断开的线段。

在边缘图像edges中,检测出长度大于accu,角度精确到1度,至少有30个像素点投票支持,最大间隙为0的线段。
返回值lines是一个N x 1 x 4的数组,N表示检测到的线段数量。每一行表示一条线段,包含这个线段的两个端点坐标(x1, y1, x2, y2)。
"""


# 创建空白图像用于绘制直线
line_img = np.zeros(img.shape, dtype=np.uint8)
# 计算图像中心点
center = (line_img.shape[1] / 2, line_img.shape[0] / 2)

# 遍历检测到的直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    # 计算斜率
    slope = (y2 - y1) / (x2 - x1)
    # 计算到中心点距离
    dist = np.abs(
        (slope * center[0] - center[1] + y1 - slope * x1) / (slope**2 + 1) ** 0.5
    )
    # 绘制直线
    cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# 显示直线检测结果
cv2.imshow("Hough Lines", line_img)
top_line = None
bottom_line = None
left_line = None
right_line = None

for line in lines:
    x1, y1, x2, y2 = line[0]

    if x1 < center[0] and x2 < center[0]:
        # 左边线
        if left_line is None or dist < left_dist:
            left_line = line
            left_dist = dist

    elif x1 > center[0] and x2 > center[0]:
        # 右边线
        if right_line is None or dist < right_dist:
            right_line = line
            right_dist = dist

    elif y1 < center[1] and y2 < center[1]:
        # 上边线
        if top_line is None or dist < top_dist:
            top_line = line
            top_dist = dist

    elif y1 > center[1] and y2 > center[1]:
        # 下边线
        if bottom_line is None or dist < bottom_dist:
            bottom_line = line
            bottom_dist = dist

# 绘制内侧框线
if left_line is not None:
    cv2.line(line_img, left_line[0][:2], left_line[0][2:], (0, 255, 0), 2)
else:
    print("Left line not detected")

if right_line is not None:
    cv2.line(line_img, right_line[0][:2], right_line[0][2:], (0, 255, 0), 2)
else:
    print("Right line not detected")

if top_line is not None:
    cv2.line(line_img, top_line[0][:2], top_line[0][2:], (0, 255, 0), 2)
else:
    print("Top line not detected")

if bottom_line is not None:
    cv2.line(line_img, bottom_line[0][:2], bottom_line[0][2:], (0, 255, 0), 2)
else:
    print("Bottom line not detected")


cv2.imshow("Inner Lines", line_img)

# 遍历所有轮廓
max_area = 0
max_cnt = None
for cnt in contours:
    # 近似轮廓形状
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    # 计算面积
    area = cv2.contourArea(cnt)

    # 如果是四边形,并且面积最大
    if len(approx) == 4 and area > max_area:
        max_area = area
        max_cnt = cnt

# 如果找到最大四边形,绘制它
if max_cnt is not None:
    cv2.drawContours(img, [max_cnt], 0, (0, 0, 255), 2)

# 保存结果
cv2.imwrite("data/rectangle_contours.jpg", img)
cv2.imshow("Aruco Codes", img)
cv2.waitKey(0)