import cv2
import screeninfo
import sys,threading
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton
# import gphoto2 as gp
class Camera:
    """open 相机的初始化
    主要用法
     frame = camera.read()
    """
    def __init__(self, id=0,camera_type="uvc"):
        self.camera_type = camera_type
        self.id=id
        if self.camera_type == "uvc":
            self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            # self.cap = cv2.VideoCapture(0)
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
    def get_camera_settings_ui(self):
        '''由于opencv会有 set5000x5000最后1080p的情况
        https://blog.csdn.net/Star_ID/article/details/126783918
        Display camera settings in a window with annotations'''

        # 获取摄像头支持的所有属性
        # enum  	cv::VideoCaptureProperties {

        # 遍历所有属性
        # for i in range(70):# 获取属性名称
            # prop_name =cv2.get(cv2.CAP_PROP_PROPERTY_NAME, i)

            # print(f"属性名称: {prop_name}")
            # 获取属性值

        def trackbar_thread():
            cv2.namedWindow("GET Camera Settings", cv2.WINDOW_NORMAL)
            # Calculate window width and height
            window_width =200 # Divide screen width by 8
            window_height = 600
            # Calculate window position
            window_x = 0  # Left edge of the screen
            window_y = 0  # Top edge of the screen
            # Create the window
            # Set window position and size
            cv2.moveWindow("GET Camera Settings", window_x, window_y)
            cv2.resizeWindow("GET Camera Settings", window_width, window_height)
            cv2.createTrackbar("Frame Width", "GET Camera Settings", 0, 5000, lambda x: None)
            cv2.createTrackbar("Frame Height", "GET Camera Settings", 0, 5000, lambda x: None)
            cv2.createTrackbar("Exposure", "GET Camera Settings", 0, 100, lambda x: None)
            cv2.createTrackbar("ISO Speed", "GET Camera Settings", 0, 100, lambda x: None)
            cv2.createTrackbar("Brightness", "GET Camera Settings", 0, 255, lambda x: None)
            cv2.createTrackbar("Contrast", "GET Camera Settings", 0, 255, lambda x: None)
            cv2.createTrackbar("Saturation", "GET Camera Settings", 0, 255, lambda x: None)
            cv2.createTrackbar("Hue", "GET Camera Settings", 0, 255, lambda x: None)
            cv2.createTrackbar("Gain", "GET Camera Settings", 0, 127, lambda x: None)
            cv2.createTrackbar("Exposure Compensation", "GET Camera Settings", -7, -1, lambda x: None)
            cv2.createTrackbar("White Balance", "GET Camera Settings", 4000, 7000, lambda x: None)
            cv2.createTrackbar("Focus", "GET Camera Settings", 0, 255, lambda x: None)

            while True:
                cv2.setTrackbarPos("Frame Width", "GET Camera Settings", int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                cv2.setTrackbarPos("Frame Height", "GET Camera Settings", int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cv2.setTrackbarPos("Exposure", "GET Camera Settings", int(self.cap.get(cv2.CAP_PROP_EXPOSURE)))
                cv2.setTrackbarPos("ISO Speed", "GET Camera Settings", int(self.cap.get(cv2.CAP_PROP_ISO_SPEED)))
                cv2.setTrackbarPos("Brightness", "GET Camera Settings", int(self.cap.get(10)))
                cv2.setTrackbarPos("Contrast", "GET Camera Settings", int(self.cap.get(11)))
                cv2.setTrackbarPos("Saturation", "GET Camera Settings", int(self.cap.get(12)))
                cv2.setTrackbarPos("Hue", "GET Camera Settings", int(self.cap.get(13)))
                cv2.setTrackbarPos("Gain", "GET Camera Settings", int(self.cap.get(14)))
                cv2.setTrackbarPos("Exposure Compensation", "GET Camera Settings", int(self.cap.get(15)))
                cv2.setTrackbarPos("White Balance", "GET Camera Settings", int(self.cap.get(17)))
                cv2.setTrackbarPos("Focus", "GET Camera Settings", int(self.cap.get(28)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        trackbar_thread = threading.Thread(target=trackbar_thread)
        trackbar_thread.start()
    def set_camera_settings(self, resolution=(1280,720), exposure=-5, brightness=100,iso=1500,contrast=50,saturation=70,hue=13,gain=50,white_balance=6500):
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
    def set_camera_settings_ui(self,cswn="Set Camera"):#     使用 OpenCV trackbars 设置摄像头参数

        def nothing(x):
            pass

      # 创建一个名为 camsettingwindowsname 的窗口
        cv2.namedWindow(cswn)
        cv2.namedWindow(cswn, cv2.WINDOW_NORMAL)

        # 获取当前摄像头的参数值

        # 创建 trackbars 用于调整增益、曝光补偿、白平衡和对焦等
        cv2.createTrackbar("Gain", cswn, 0, 100, nothing)
        cv2.createTrackbar("White Balance", cswn, 4000, 7000, nothing)
        cv2.createTrackbar("Frame Width", cswn, 0, 5000, lambda x: None)
        cv2.createTrackbar("Frame Height", cswn, 0, 5000, lambda x: None)
        cv2.createTrackbar("Exposure", cswn, -10, 10, lambda x: None)
        cv2.createTrackbar("ISO Speed", cswn, 0, 100, lambda x: None)
        cv2.createTrackbar("Brightness", cswn, 0, 255, lambda x: None)
        cv2.createTrackbar("Contrast", cswn, 0, 255, lambda x: None)
        cv2.createTrackbar("Saturation", cswn, 0, 255, lambda x: None)
        cv2.createTrackbar("Hue", cswn, 0, 255, lambda x: None)

        cv2.setTrackbarPos("Gain", cswn, int(self.cap.get(14)))
        cv2.setTrackbarPos("White Balance", cswn, int(self.cap.get(17)))
        cv2.setTrackbarPos("Frame Width", cswn, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cv2.setTrackbarPos("Frame Height", cswn, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cv2.setTrackbarPos("Exposure", cswn, int(self.cap.get(cv2.CAP_PROP_EXPOSURE)))
        cv2.setTrackbarPos("ISO Speed", cswn, int(self.cap.get(cv2.CAP_PROP_ISO_SPEED)))
        cv2.setTrackbarPos("Brightness", cswn, int(self.cap.get(10)))
        cv2.setTrackbarPos("Contrast", cswn, int(self.cap.get(11)))
        cv2.setTrackbarPos("Saturation", cswn, int(self.cap.get(12)))
        cv2.setTrackbarPos("Hue", cswn, int(self.cap.get(13)))
        def trackbar_thread():
            while True:
                # 获取当前的 trackbar 值
                gain = cv2.getTrackbarPos("Gain", cswn)
                white_balance = cv2.getTrackbarPos("White Balance", cswn)
                frame_width = cv2.getTrackbarPos("Frame Width", cswn)
                frame_height = cv2.getTrackbarPos("Frame Height", cswn)
                exposure = cv2.getTrackbarPos("Exposure", cswn)
                iso_speed = cv2.getTrackbarPos("ISO Speed", cswn)
                brightness = cv2.getTrackbarPos("Brightness", cswn)
                contrast = cv2.getTrackbarPos("Contrast", cswn)
                saturation = cv2.getTrackbarPos("Saturation", cswn)
                hue = cv2.getTrackbarPos("Hue", cswn)
                # self.cap.set(14, gain)
                self.cap.set(17, white_balance)# white_balance  min: 4000, max: 7000, increment:1
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  frame_height)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)# exposure       min: -7  , max: -1  , increment:1
                self.cap.set(cv2.CAP_PROP_ISO_SPEED, 1500)
                self.cap.set(10, brightness) # brightness     min: 0   , max: 255 , increment:1
                self.cap.set(11, contrast   ) # contrast       min: 0   , max: 255 , increment:1
                self.cap.set(12, saturation   ) # saturation     min: 0   , max: 255 , increment:1
                self.cap.set(13, hue   ) # hue
        trackbar_thread = threading.Thread(target=trackbar_thread)
        trackbar_thread.start()

    def read(self,ispreview="no"):
        if self.camera_type == "uvc":
            ret, frame = self.cap.read()
            if ispreview=="preview":
                cv2.imshow('This is a picture taken by myCamera just now', frame)
                # key = cv2.waitKey(0)#等待0ms读入键盘
                # if key & 0xFF == ord('q'):
                #     preview = False
                #     cv2.destroyWindow("This is a picture taken by myCamera just now")
        return ret,frame



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
            self.id=selected_screen_id
            self.width=self.monitors[selected_screen_id].width
            self.height=self.monitors[selected_screen_id].height
            return selected_screen_id
        return 0

    def cv_create_window(self, windowname, resizewindowsize=None, fullscreen=False, id=None):
        if id is None:
            id = self.id
        # 控制窗口显示
        cv2.namedWindow(windowname, cv2.WND_PROP_FULLSCREEN)
        screen = self.monitors[id]
        width, height = screen.width, screen.height
        if fullscreen:
            cv2.namedWindow(windowname, cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty(windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(windowname, screen.x - 1, screen.y - 1)  # 为了确保窗口不会被放置在屏幕的边缘之外，会在指定的坐标上减去一个像素。这样可以确保窗口不会被裁剪或部分显示在屏幕边缘之外。
        elif resizewindowsize is not None:
            cv2.resizeWindow(windowname, resizewindowsize[0], resizewindowsize[1])
            cv2.moveWindow(windowname, screen.x + width // 4, screen.y + height // 4)

    def show_image(self, windowname,img):
        cv2.imshow(windowname, img)



if __name__ == "__main__":
    # screen = Screen()
    # print(screen.guiselect())
    cam = Camera("uvc")
    cam.read("preview")
    resolution = (1200,700)
    exposure = -5
    brightness =100
    #注意此set会改变画面参数 需要打开windows相机才可恢复正常
    cam.get_camera_settings_ui()
    # cam.set_camera_settings(resolution, exposure,brightness)
    # cam.set_camera_settings_ui()


    while True:
        ret,frame = cam.read()
        if ret:
            cv2.imshow("Image", frame)
        # 等待按键按下
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
