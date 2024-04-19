import cv2
import numpy as np
import sys,os  # Import sys module for path manipulation
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import method.camera_screen as cs
class PatternMarker:
    """
    基类，用于表示图案标记。

    Attributes:
    markersX: X轴上标记的数量
    markersY: Y轴上标记的数量
    markerLength: 标记的长度，单位是像素
    markerSeparation: 每个标记之间的间隔，单位像素
    margins: 标记与边界之间的间隔
    borderBits: 标记的边界所占的bit位数
    """

    def __init__(self, markersX, markersY, markerLength, markerSeparation,margins):
        self.markersX = markersX
        self.markersY = markersY
        self.markerLength = markerLength
        self.markerSeparation = markerSeparation
        self.margins = margins
        self.width = (
            self.markersX * (self.markerLength + self.markerSeparation)
            - self.markerSeparation
            + 2 * self.margins
        )
        self.height = (
            self.markersY * (self.markerLength + self.markerSeparation)
            - self.markerSeparation
            + 2 * self.margins
        )

    def get_total_markers(self):
        """
        获取图案标记的总数。
        """
        return self.markersX * self.markersY

    def get_marker_corners(self, markerX, markerY):
        """
        获取指定图案标记的四个角点坐标。

        Args:
            markerX (int): 图案标记在横向上的索引。
            markerY (int): 图案标记在纵向上的索引。

        Returns:
            list[tuple]: 四个角点坐标的列表。
        """
        x0 = markerX * self.markerSeparation + self.margins
        y0 = markerY * self.markerSeparation + self.margins
        x1 = x0 + self.markerLength
        y1 = y0 + self.markerLength
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]



class Chess(PatternMarker):
    """
    棋盘格图案标记类。

    Attributes:
        markersX (int): 图案标记在横向上的个数。
        markersY (int): 图案标记在纵向上的个数。
        markerLength (float): 图案标记的长度。
        markerSeparation (float): 图案标记之间的间距。
    """

    def __init__(self, markersX, markersY, markerLength, margins):
        super().__init__(markersX, markersY, markerLength,0, margins)
        #self.markerSeparation = 0  # 棋盘格图案标记没有边缘空白区域

    def get_marker_corners(self, markerX, markerY):
        """
        获取指定棋盘格图案标记的四个角点坐标。

        Args:
            markerX (int): 棋盘格图案标记在横向上的索引。
            markerY (int): 棋盘格图案标记在纵向上的索引。

        Returns:
            list[tuple]: 四个角点坐标的列表。
        """
        x0 = markerX * self.markerSeparation
        y0 = markerY * self.markerSeparation
        x1 = x0 + self.markerLength
        y1 = y0 + self.markerLength
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    def generateChessboard(self,chesspath="data/chessboard.jpg"):
        # Insert code here to generate the chessboard image
        chessboard = np.zeros((self.height, self.width), dtype=np.uint8)
        for i in range(0, self.height, self.markerLength ):
            for j in range(0, self.width, self.markerLength ):
                if (i // (self.markerLength )) % 2 == (j // (self.markerLength )) % 2:
                    chessboard[i:i + self.markerLength, j:j + self.markerLength] = 255
        #cv2.imwrite(chesspath, chessboard)
        cv2.imshow("Chessboard", chessboard)
        cv2.waitKey(0)

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
class Aruco(PatternMarker):
    """
    Artag图案标记类。

    Attributes:
        markersX (int): 图案标记在横向上的个数。
        markersY (int): 图案标记在纵向上的个数。
        markerLength (float): 图案标记的长度。
        markerSeparation (float): 图案标记之间的间距。
        margins (float): 图案标记边缘的空白区域。
    """

    def __init__(self, markersX, markersY, markerLength, markerSeparation):
        super().__init__(markersX, markersY, markerLength, markerSeparation)


    def generate_aruco_codes(self,arcuopath="data/aruco_codes.jpg"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # 使用4x4的50个码字典
        def generate_aruco_image(dictionary, marker_size, marker_id):
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
            aruco_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            return aruco_img

        img1 = generate_aruco_image(aruco_dict, self.markerLength, 0)
        img2 = generate_aruco_image(aruco_dict, self.markerLength, 1)
        img3 = generate_aruco_image(aruco_dict, self.markerLength, 2)
        img4 = generate_aruco_image(aruco_dict, self.markerLength, 3)
        img = np.ones((self.height, self.width), dtype=np.uint8)*255
        img[0:self.markerLength, 0:self.markerLength] = img1
        img[0:self.markerLength, self.width-self.markerLength:self.width] = img2
        img[self.height-self.markerLength:self.height, 0:self.markerLength] = img3
        img[self.height-self.markerLength:self.height, self.width-self.markerLength:self.width] = img4

        cv2.imwrite(arcuopath, img)
        cv2.imshow("Aruco Codes", img)
        cv2.waitKey(0)

    def chesscalibrate(self,chesscampath='data\chessboard.jpg'):
        # 阈值
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # print(cv2.TERM_CRITERIA_EPS,'',cv2.TERM_CRITERIA_MAX_ITER)
        img = cv2.imread(chesscampath)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.markersX ,self.markersY))
        # 画出corners
        for corner in corners:
            cv2.circle(img, (corner[0][0], corner[0][1]), 5, (255, 0, 0), -1)
        # 如果找到足够点对，将其存储起来
        corners3 = np.array([])
        if ret == True:
            #精确找到角点坐标
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            corners3 = np.array([corners[0],corners[10],corners[77],corners[87]])
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (2,2), corners3, ret)
            print(corners3)
            # cv2.imshow('findCorners',img)
            # cv2.waitKey()
        # cv2.destroyAllWindows()

        # print('四个点的坐标是:\n',corners3)
        point = np.reshape(corners3,(4,2))
        print(point)

        dst = np.array([[200, 200],
                        [400, 200],
                        [200, 400],
                        [400, 400]], dtype = "float32")
        M = cv2.getPerspectiveTransform(point, dst)
        print('变换矩阵是', M)
        warped = cv2.warpPerspective(img, M, (1000, 1000))

        img = cv2.resize(img,(400,400))
        cv2.imshow('img',img)
        cv2.imshow('fin',warped)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def calibrateProjector(self):
        # Insert code here to calibrate the projector
        camera = cs.Camera("uvc")
        frame=camera.read()

# 棋盘格尺寸





if __name__ == '__main__':
        # 1.创建显示窗口
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 1075, 900)
    # 棋盘格参数
    corners_vertical = 8    # 纵向角点个数;
    corners_horizontal = 11  # 纵向角点个数;


    resolution=(1280,720)
    test=PatternMarker(5, 3, 100, 0)
    test.generateChessboard()
    test.chesscalibrate()
    test.generate_aruco_codes()
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

    # 示例用法
    chess_pattern = Chess(5, 4, 0.1, 0.12)
    print(chess_pattern.get_total_markers())  # 输出：20
    print(chess_pattern.get_marker_corners(2, 1))  # 输出：[(0.24, 0.12), (0.36, 0.12), (0.36, 0.24), (0.24, 0.24)]

    aruco_pattern = Aruco(6, 6, 0.05, 0.06)
    print(aruco_pattern.get_total_markers())  # 输出：36
    print(aruco_pattern.get_marker_corners(3, 2))  # 输出：[(0.21, 0.18

