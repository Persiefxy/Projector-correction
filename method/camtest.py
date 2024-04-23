import camera_screen as cs
import cv2
cam = cs.Camera("uvc")
cam = cs.Camera(True)
cam.set_camera_settings((1280,720))
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('r'):
        print("拍摄")
        break
ret, cam_img = cam.read()  # 从相机读取一帧
if not ret:
    print('未能获取图像...')

# preview = cv2.resize(cam_img, (0, 0), fx=0.5, fy=0.5)
filename = "./data/aruco_new.png"
cv2.imwrite(filename, cam_img)


