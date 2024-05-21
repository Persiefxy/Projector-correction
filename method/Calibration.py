import cv2
import numpy as np
# import sys,os  # Import sys module for path manipulation
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import camera_screen as cs
def dict1(): #为了适配之前的aruco数据而妥协的一个转置
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    myByteList = []
    for idx, code in enumerate(aruco_dict.bytesList):
        #遍历现有Aruco字典（aruco_dict.bytesList）中的每个编码（code）
        code = code[np.newaxis, :, :]
        # 将当前的code通过np.newaxis增加一个维度，以适配cv2.aruco.Dictionary_getBitsFromByteList函数的输入要求。
        bits = cv2.aruco.Dictionary_getBitsFromByteList(code, 4)
        #cv2.aruco.Dictionary_getBitsFromByteList函数将字节列表转换为位矩阵（bits）。
        bits = cv2.flip(bits, 1)
        #使用cv2.flip函数沿着水平方向（参数为1）翻转位矩阵，这可能是为了满足特定的标记识别要求或纠正方向。
        code = cv2.aruco.Dictionary_getByteListFromBits(bits)
        # 将翻转后的位矩阵(bits)重新转换为字节列表(code)，以便后续使用。
        myByteList.append(code[0])
        #将转换后的字节列表添加到myByteList中。
    myByteList = np.stack(myByteList, axis=0)
    aruco_dict = cv2.aruco.Dictionary(myByteList, 4) # 使用4x4的50个码字典  +转置
    return aruco_dict
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
        x0 = markerX * (self.markerSeparation+ self.markerLength)+ self.margins
        y0 = markerY * (self.markerSeparation+ self.markerLength) + self.margins
        x1 = x0 + self.markerLength
        y1 = y0 + self.markerLength
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    def get_marker_anchors(self, markerX, markerY):
        """
            获取指定图案标记的中心坐标。

            Args:
                markerX (int): 图案标记在横向上的索引。
                markerY (int): 图案标记在纵向上的索引。
            Returns:
                tuple: 中心坐标 (x, y)。
        """
        corners = self.get_marker_corners(markerX, markerY)
        center = np.mean(corners, axis=0)
        return center
class Chess(PatternMarker):
    """
    棋盘格图案标记类。

    Attributes:
        markersX (int): 图案标记在横向上的个数。
        markersY (int): 图案标记在纵向上的个数。
        markerLength (float): 图案标记的长度。
        markerSeparation (float): 图案标记之间的间距。
    """

    def __init__(self, pattern_size, square_size):
        super().__init__(pattern_size[0], pattern_size[1], square_size,0, 0)
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
        # cv2.imshow("Chessboard", chessboard)
        # cv2.waitKey(0)
        return chessboard


    def find_corners_sb(self,img,pattern_size):
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


    def find_corners(self,img,pattern_size):
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
    def chesscalibrate(self,corners3,img):
        # 阈值
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

    # def generate_aruco4_codes(self,arcuopath="data/aruco_codes.jpg"):

    #     def generate_aruco_image(dictionary, marker_size, marker_id):
    #         aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    #         aruco_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    #         aruco_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    #         return aruco_img
    #     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    #     img1 = generate_aruco_image(aruco_dict, self.markerLength, 0)
    #     img2 = generate_aruco_image(aruco_dict, self.markerLength, 1)
    #     img3 = generate_aruco_image(aruco_dict, self.markerLength, 2)
    #     img4 = generate_aruco_image(aruco_dict, self.markerLength, 3)
    #     img = np.ones((self.height, self.width), dtype=np.uint8)*255
    #     img[0:self.markerLength, 0:self.markerLength] = img1
    #     img[0:self.markerLength, self.width-self.markerLength:self.width] = img2
    #     img[self.height-self.markerLength:self.height, 0:self.markerLength] = img3
    #     img[self.height-self.markerLength:self.height, self.width-self.markerLength:self.width] = img4
    #     return img
    def generate_aruco15_codes(self):
        borderBits = 1 #标记的边界所占的bit位数
        board = cv2.aruco.GridBoard((self.markersX, self.markersY),float(self.markerLength),float(self.markerSeparation), dict1())
        img = cv2.aruco.drawPlanarBoard(board,(self.width,self.height),self.margins,borderBits=borderBits)
        return img
    def generate_corners_anchors(self):
        anchors={}
        ids=[]
        corners=[]
        for y in range(self.markersY):
            for x in range(self.markersX):
                marker_id = x + y * self.markersX
                marker_corners=self.get_marker_corners(x, y)
                ids.append(marker_id)
                corners.append(marker_corners)
                marker_anchors=self.get_marker_anchors(x, y)
                anchors[marker_id]=marker_anchors
        return corners,ids,anchors

class dotmatrix(PatternMarker):
    def generate_dot_matrix(self, dot_size=10, dot_color=(255, 255, 255)):
        """
        生成点阵图像。

        Args:
            dot_size (int): 每个点的大小（像素）。
            dot_color (tuple): 点的颜色，格式为(R, G, B)。

        Returns:
            numpy.ndarray: 生成的点阵图像。
        """
        # 创建一个黑色背景的图像
        dot_matrix = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 在图像上绘制点阵
        for y in range(0, self.markersY, dot_size * 2):
            for x in range(0, self.markersX, dot_size * 2):
                dot_matrix[y:y+dot_size, x:x+dot_size] = dot_color
        return dot_matrix


if __name__ == '__main__':
    # 棋盘格参数
    corners_vertical = 8    # 纵向角点个数;
    corners_horizontal = 11  # 纵向角点个数;
    board_size= (corners_vertical, corners_horizontal)
    pattern_size = (corners_vertical-2, corners_horizontal-2)
  # 棋盘格尺寸
    square_size = 50  # 每个棋盘格的大小

    resolution=(1280,720)
    test=Chess(corners_vertical, corners_horizontal, square_size, 0)
    image = test.generateChessboard()




    # 生成棋盘格图像
    # 将灰度图像转换为彩色图像
    img=cv2.imread(r"E:\5graduateproject\muilt-Projector-correction\data\OpenCVtabletorecognize1.jpeg")

    test.find_corners_sb(img,pattern_size)
    cv2.imshow('img', img)

    # 显示棋盘格图像
    cv2.waitKey(0)

    # 示例用法
    chess_pattern = Chess(5, 4, 10, 10)
    print(chess_pattern.get_total_markers())  # 输出：20
    print(chess_pattern.get_marker_corners(2, 1))  # 输出：[(0.24, 0.12), (0.36, 0.12), (0.36, 0.24), (0.24, 0.24)]


    aruco_pattern = Aruco(5,3,100,20,20)
    print(aruco_pattern.get_total_markers())  # 输出：36
    print(aruco_pattern.get_marker_corners(3, 2))  # 输出：[(0.21, 0.18

