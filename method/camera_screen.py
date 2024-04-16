import cv2
import screeninfo
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton

# import gphoto2 as gp
class Camera:
    """open 相机的初始化
    主要用法
     frame = camera.read()
    """
    def __init__(self, camera_type):
        self.camera_type = camera_type

    def open(self,use_dshow=True):
        if self.camera_type == "uvc":
            if use_dshow:
                self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
              print("无法打开摄像头")
              exit(0)
        # elif self.camera_type == "canon":
        #     self.cam =gp.Camera()
        #     self.cam.init()
        else:
            raise ValueError("Unknown camera type")
    def get_camera_settings(self):
        camera_settings = {
            "gain": self.cap.get(14),
            "white_balance": self.cap.get(17),
            "frame_width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "frame_height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
            "iso_speed": self.cap.get(cv2.CAP_PROP_ISO_SPEED),
            "brightness": self.cap.get(10),
            "contrast": self.cap.get(11),
            "saturation": self.cap.get(12),
            "hue": self.cap.get(13)
        }
        return camera_settings
    def set_camera_settings(self, resolution=(1280,720), exposure=-5, brightness=120,iso=1500,contrast=50,saturation=70,hue=13,gain=50,white_balance=6500):
        """https://techoverflow.net/2018/12/18/how-to-set-cv2-videocapture-image-size/"""
        if self.camera_type == "uvc":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            # if not self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure):
            #     raise ValueError("Failed to set exposure value")
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.cap.set(cv2.CAP_PROP_ISO_SPEED, iso)
            self.cap.set(10, brightness) # brightness     min: 0   , max: 255 , increment:1
            self.cap.set(11, 50   ) # contrast       min: 0   , max: 255 , increment:1
            self.cap.set(12, 70   ) # saturation     min: 0   , max: 255 , increment:1
            self.cap.set(13, 13   ) # hue
            self.cap.set(14, 50   ) # gain           min: 0   , max: 127 , increment:1
            self.cap.set(17, 6500 ) # white_balance  min: 4000, max: 7000, increment:1
            #https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html
        else:
            raise ValueError("Getting camera settings is not supported for this camera type")
    def get_set_camera_settings_ui(self):#     使用 OpenCV trackbars 设置摄像头参数

        def nothing(x):
            pass


      # 创建一个名为 "Camera Settings" 的窗口
        cv2.namedWindow("Camera Settings")
        cv2.namedWindow("Camera Settings", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Camera Settings", 400, 900)
        # 获取当前摄像头的参数值
  
        # 创建 trackbars 用于调整增益、曝光补偿、白平衡和对焦等
        cv2.createTrackbar("Gain", "Camera Settings", 0, 100, nothing)
        cv2.createTrackbar("White Balance", "Camera Settings", 4000, 7000, nothing)
        cv2.createTrackbar("Frame Width", "Camera Settings", 0, 5000, lambda x: None)
        cv2.createTrackbar("Frame Height", "Camera Settings", 0, 5000, lambda x: None)
        cv2.createTrackbar("Exposure", "Camera Settings", -10, 10, lambda x: None)
        cv2.createTrackbar("ISO Speed", "Camera Settings", 0, 100, lambda x: None)
        cv2.createTrackbar("Brightness", "Camera Settings", 0, 255, lambda x: None)
        cv2.createTrackbar("Contrast", "Camera Settings", 0, 255, lambda x: None)
        cv2.createTrackbar("Saturation", "Camera Settings", 0, 255, lambda x: None)
        cv2.createTrackbar("Hue", "Camera Settings", 0, 255, lambda x: None)
        
        cv2.setTrackbarPos("Gain", "Camera Settings", int(self.cap.get(14)))
        cv2.setTrackbarPos("White Balance", "Camera Settings", int(self.cap.get(17)))
        cv2.setTrackbarPos("Frame Width", "Camera Settings", int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cv2.setTrackbarPos("Frame Height", "Camera Settings", int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cv2.setTrackbarPos("Exposure", "Camera Settings", int(self.cap.get(cv2.CAP_PROP_EXPOSURE)))
        cv2.setTrackbarPos("ISO Speed", "Camera Settings", int(self.cap.get(cv2.CAP_PROP_ISO_SPEED)))
        cv2.setTrackbarPos("Brightness", "Camera Settings", int(self.cap.get(10)))
        cv2.setTrackbarPos("Contrast", "Camera Settings", int(self.cap.get(11)))
        cv2.setTrackbarPos("Saturation", "Camera Settings", int(self.cap.get(12)))
        cv2.setTrackbarPos("Hue", "Camera Settings", int(self.cap.get(13)))

        while True:
            # 获取当前的 trackbar 值
            gain = cv2.getTrackbarPos("Gain", "Camera Settings")
            white_balance = cv2.getTrackbarPos("White Balance", "Camera Settings")
            frame_width = cv2.getTrackbarPos("Frame Width", "Camera Settings")
            frame_height = cv2.getTrackbarPos("Frame Height", "Camera Settings")
            exposure = cv2.getTrackbarPos("Exposure", "Camera Settings")
            iso_speed = cv2.getTrackbarPos("ISO Speed", "Camera Settings")
            brightness = cv2.getTrackbarPos("Brightness", "Camera Settings")
            contrast = cv2.getTrackbarPos("Contrast", "Camera Settings")
            saturation = cv2.getTrackbarPos("Saturation", "Camera Settings")
            hue = cv2.getTrackbarPos("Hue", "Camera Settings")
                        # 设置摄像头的参数
            self.cap.set(14, gain)
            self.cap.set(17, white_balance)# white_balance  min: 4000, max: 7000, increment:1
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  frame_height)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)# exposure       min: -7  , max: -1  , increment:1
            self.cap.set(cv2.CAP_PROP_ISO_SPEED, 1500)
            self.cap.set(10, brightness) # brightness     min: 0   , max: 255 , increment:1
            self.cap.set(11, contrast   ) # contrast       min: 0   , max: 255 , increment:1
            self.cap.set(12, saturation   ) # saturation     min: 0   , max: 255 , increment:1
            self.cap.set(13, hue   ) # hue
            ret, frame = self.cap.read()
            if not ret:
                return
            cv2.imshow("Camera", frame)
            # 等待用户按下 Esc 键退出
            if cv2.waitKey(1) == 27:
                break
        # 关闭窗口
        cv2.destroyAllWindows()

    def read(self,preview=False):
        if not hasattr(self, 'cap'):
            self.open()
        if self.camera_type == "uvc":
            ret, frame = self.cap.read()
            assert ret, "Error reading frame from camera"

            if preview:
                cv2.imshow('Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.preview = False

            return frame



    def release(self):
        if self.camera_type == "uvc":
            self.cap.release()
        elif self.camera_type == "canon":
            self.cam.exit()
        else:
            raise ValueError("Unknown camera type")
class Image:
    def __init__(self, frame):
        self.frame = frame
    def show(self):
        cv2.imshow("Image", self.frame)


class Screen:
    # 调用投影仪
    def __init__(self):
        self.monitors = screeninfo.get_monitors()
    def guiselect(self):
        screen_selected = False
        selected_screen_id=0
        def onButtonClicked():
            nonlocal screen_selected,selected_screen_id
            screen_selected= True
            selected_screen_id = combo.currentIndex()
            app.quit()
        app = QApplication(sys.argv)
        window = QWidget()
        window.setWindowTitle('Select Screen')
        options = [f"{monitor.name} - Resolution: {monitor.width}x{monitor.height} - ID: {i}" for i,monitor in enumerate(self.monitors)]
        combo = QComboBox(window)
        combo.addItems(options)
        combo.setCurrentIndex(0)
        button = QPushButton('Select', window)
        button.clicked.connect(onButtonClicked)
        combo.move(50, 50)
        button.move(450, 50)
        window.setGeometry(100, 100, 600, 200)
        window.show()
        app.exec_()
        if screen_selected:
            return selected_screen_id

    def move_window(self,id,windowname):
        #控制显示屏幕
        screen = self.monitors[id]
        width, height = screen.width, screen.height
        cv2.namedWindow(windowname, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(windowname, screen.x - 1, screen.y - 1)#为了确保窗口不会被放置在屏幕的边缘之外，会在指定的坐标上减去一个像素。这样可以确保窗口不会被裁剪或部分显示在屏幕边缘之外。
        cv2.setWindowProperty(
            windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )   

    def show_image(self, windowname,img):
        cv2.imshow(windowname, img)



if __name__ == "__main__":
    screen = Screen()
    print(screen.guiselect())
    camera = Camera("uvc")
    resolution = (800,700)
    exposure = -5
    brightness =1
    #注意此set会改变画面参数 需要打开windows相机才可恢复正常
    camera.open()
    camera.set_camera_settings(resolution, exposure,brightness)
    camera.get_set_camera_settings_ui()

