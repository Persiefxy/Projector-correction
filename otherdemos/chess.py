import cv2
import numpy as np

# 棋盘格尺寸

# 生成棋盘格图像
def generate_chessboard(pattern_size,square_size=20):
    """
    生成棋盘格图像

    参数：
        pattern_size：棋盘格尺寸，元组形式，例如 (7, 8)

    返回值：
        棋盘格图像 (ndarray)
    """
    
    # 创建一个空图像
    image = np.zeros((pattern_size[0] * square_size, pattern_size[1] * square_size), dtype=np.uint8)
    # 填充棋盘格
    for i in range(pattern_size[0]):
        for j in range(pattern_size[1]):
            if (i + j) % 2 == 0:
                color = 255  # 白色
            else:
                color = 0  # 黑色
            image[i * square_size : (i + 1) * square_size, j * square_size : (j + 1) * square_size] = color
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image



def find_corners_sb(img,pattern_size):
    """
    查找棋盘格角点函数 SB升级款
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    if ret:
        # 显示角点
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        return corners


def find_corners(img,pattern_size):
    """
    查找棋盘格角点函数
    :param img: 处理原图
    """
    # 查找棋盘格 角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_FILTER_QUADS)
    if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 显示角点
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        return corners
    return -1

if __name__ == '__main__':
        # 1.创建显示窗口
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 1075, 900)
    # 棋盘格参数
    corners_vertical = 8    # 纵向角点个数;
    corners_horizontal = 11  # 纵向角点个数;


    board_size= (corners_vertical+2, corners_horizontal+2)
    pattern_size = (corners_vertical, corners_horizontal)
    # 棋盘格尺寸
    square_size = 50  # 每个棋盘格的大小
    # 生成棋盘格图像
    image = generate_chessboard(board_size, square_size)
    # 将灰度图像转换为彩色图像
    img=cv2.imread("data/OpenCVtabletorecognize1.jpeg")

    find_corners_sb(img,(7,5))
    cv2.imshow('img', img)
    
    # 显示棋盘格图像
    cv2.waitKey(0)
