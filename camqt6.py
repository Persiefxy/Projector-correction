import sys

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QPushButton, QWidget
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('pyqt6显示opencv获取的摄像头图像')
        self.btn_camera = QPushButton('打开摄像头')  # 控制摄像头的状态
        self.lbl_img = QLabel('显示摄像头图像')  # 创建标签控件来显示摄像头的图像, 标签的大小由QGridLayout的布局来决定
        self.lbl_img.setStyleSheet('border: 1px solid black;')  # 给标签设置黑色边框
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 让标签要显示的内容居中
        self.lbl_img.setMinimumSize(640, 480)  # 宽和高保持和摄像头获取的默认大小一致
        self.btn_camera.clicked.connect(self.btn_camera_click)
        top_widget = QWidget()
        grid = QGridLayout()
        grid.addWidget(self.lbl_img, 0, 0, Qt.AlignmentFlag.AlignTop)  # 放置顶部
        grid.addWidget(self.btn_camera, 1, 0, Qt.AlignmentFlag.AlignBottom)  # 放置底部
        top_widget.setLayout(grid)
        self.setCentralWidget(top_widget)

        self.center_win()  # 居中显示主窗口

        self.is_open_camera = False  # 是否打开了摄像头标志位
        self.video_cap = None
        self.camera_timer = QtCore.QTimer(self)  # 创建读取摄像头图像的定时器
        self.camera_timer.timeout.connect(self.play_camera_video)

    def center_win(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def btn_camera_click(self):
        if not self.is_open_camera: # 按下 打开摄像头 按钮
            self.video_cap = cv2.VideoCapture(0)  # 打开默认摄像头（索引为0）
            print('camera fps:', self.video_cap.get(cv2.CAP_PROP_FPS))
            # 每个20毫秒获取一次摄像头的图像进行刷新, 具体设置多少合适, 可以参考你的摄像头帧率cv2.CAP_PROP_FPS,
            # 刷新频率设置一个小于 1000 / cv2.CAP_PROP_FPS 的值即可
            self.camera_timer.start(20)
            self.is_open_camera = True
            self.btn_camera.setText('关闭摄像头')
        else:  # 按下 关闭摄像头 按钮
            self.camera_timer.stop()
            self.video_cap.release()
            self.video_cap = None
            self.lbl_img.clear()
            self.btn_camera.setText('打开摄像头')
            self.is_open_camera = False

    def play_camera_video(self):
        if self.is_open_camera:
            # ret, frame = self.video_cap.read()  # 读取视频流的每一帧
            self.video_cap.grab()
            ret, frame = self.video_cap.retrieve()  # 读取视频流的每一帧
            if ret:
                height, width, channel = frame.shape  # 获取图像高度、宽度和通道数, 通常为为640x480x3
                # opencv获取的图像默认BGR格式
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 转换BRG 到 RGB
                # 或者
                # 将OpenCV格式转换成QImage格式, 需要进行颜色通道交换(.rgbSwapped())
                img = QImage(frame.data, width, height, QImage.Format.Format_RGB888)
                img = img.rgbSwapped()
                # 或者
                # img = QImage(frame.data, width, height, QImage.Format.Format_RGB888)
                # img.rgbSwap()
                # 或者
                # img = QImage(frame.data, width, height, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(img)  # 从QImage生成QPixmap对象
                #
                lbl_width = self.lbl_img.size().width()  # 通过size()获取图像标签的真实的宽度
                lbl_height = self.lbl_img.size().height()  # 通过size()获取图像标签的真实的高度
                # 按照图像标签的真实的宽和高进行缩放，但保持摄像头的宽和高的比例
                pixmap = QPixmap(pixmap).scaled(
                    self.lbl_img.width(), self.lbl_img.height(),
                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
                self.lbl_img.setPixmap(pixmap)  # 在标签上显示图片


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())