# 换源
bash or cmd里面 不要在vscode里面
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装opencv
pip install opencv-python==4.7.0.68
pip install opencv-contrib-python==4.7.0.68
pip install scipy screeninfo pyqt5
# 更改摄像头查找
cv2.VideoCapture(
linux下插入摄像头前后ls /dev/video* 看多了那个 更改设备树后可能会不一样？
/dev/video2  /dev/video3  /dev/video4  /dev/video5

# 保存拍摄预览等
## ipynb plt
   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
## 捕获
cam = cv2.VideoCapture(0)  # 假设0是相机的索引，根据实际情况调整
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)



## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

## 从相机读取一帧
    ret, cam_img = cam.read()
    if not ret:
        print('未能获取图像...')
        continue  # 如果图像读取失败，继续下一次循环
    # 对读取到的图像进行缩放，显示预览
    preview = cv2.resize(cam_img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Preview", preview)
    key = cv2.waitKey(1) & 0xFF
    # 按空格键保存图像
    if key == 32:
        i_code = 0
        cv2.imshow('GrayCode', pattern_images[i_code])
        cv2.waitKey(500)
        capture_mode = True
        continue
    # 按ESC键退出
    elif key == 27:
        break

 ## 保存图像
        cv2.imwrite(arcuopath, img)
        cv2.imshow("Aruco Codes", img)
        cv2.waitKey(0)

        filename = f"{capture_dir}/gc_{i_code:04d}.png"
        cv2.imwrite(filename, cam_img)
        print(f"Saved {filename}")
        i_code += 1
        if i_code >= len(pattern_images):
            break
        cv2.imshow('GrayCode', pattern_images[i_code])
        cv2.waitKey(500)
#释放相机资源并关闭所有OpenCV窗口
cam.release()
cv2.destroyAllWindows()

# 屏幕
monitors = screeninfo.get_monitors()
screen = monitors[displaynum]
width, height = screen.width, screen.height
cv2.namedWindow("GrayCode", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("GrayCode", screen.x - 1, screen.y - 1)
cv2.setWindowProperty("GrayCode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('GrayCode', grid_img)
