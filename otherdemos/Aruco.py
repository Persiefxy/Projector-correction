import cv2
import numpy as np
import sys,os  # Import sys module for path manipulation
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) 
import method.camera_screen as cs

class Calibration:
    def __init__(self, markersX, markersY, markerLength, markerSeparation):
        self.markersX = markersX  #横向个数
        self.markersY = markersY  #纵向个数
        self.markerLength = markerLength
        self.markerSeparation = markerSeparation
        self.margins = markerSeparation
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

    def generateChessboard(self,chesspath="data/chessboard.jpg"):
        # Insert code here to generate the chessboard image
        chessboard = np.zeros((self.height, self.width), dtype=np.uint8)
        for i in range(0, self.height, self.markerLength + self.markerSeparation):
            for j in range(0, self.width, self.markerLength + self.markerSeparation):
                if (i // (self.markerLength + self.markerSeparation)) % 2 == (j // (self.markerLength + self.markerSeparation)) % 2:
                    chessboard[i:i + self.markerLength, j:j + self.markerLength] = 255
        cv2.imwrite(chesspath, chessboard)
        cv2.imshow("Chessboard", chessboard)
        cv2.waitKey(0)

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
        camera = Camera.Camera("uvc")
        frame=camera.read()
        

if __name__ == "__main__":
        resolution=(1280,720)
        test=Calibration(5, 3, 100, 0)
        test.generateChessboard()
        test.chesscalibrate()
        test.generate_aruco_codes()

