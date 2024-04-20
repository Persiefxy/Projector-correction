import os
import cv2
import numpy as np
import method.camera_screen as cs
import method.Calibration as cb
def copy(size,r=3,c=5):
    dx=size[0]/(c-1)
    dy=size[1]/(r-1)
    x=0
    y=0
    re=[]
    for i in range(r):
        for j in range(c):
            re.append((x,y))
            x=x+dx
        y=y+dy
        x =0
    return re



# Calculate the anchor positions for each ArUco marker

#re = copy((1199,599))
aruco1 = cb.Aruco(5,3,100,20,20)
generated_aruco_img=aruco1.generate_aruco15_codes()
corners,ids,anchors=aruco1.generate_corners_anchors()
re=[]
for key, data in anchors.items():
    re.append(data)
#调用投影仪
myscreen=cs.Screen()
monitors = myscreen.monitors
myscreen = monitors[myscreen.guiselect()]
width, height = myscreen.width, myscreen.height

cv2.namedWindow("ARTag", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("ARTag", myscreen.x - 1, myscreen.y - 1)
cv2.setWindowProperty("ARTag", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('ARTag', generated_aruco_img)
cv2.waitKey(0)
#调用摄像头
cam = cs.Camera()
# cam.set_camera_settings((1280,720))
ret, cam_img = cam.read()  # 从相机读取一帧
if not ret:
    print('未能获取图像...')

preview = cv2.resize(cam_img, (0, 0), fx=0.5, fy=0.5)
filename = r"./data/aruco_new.png"
cv2.imwrite(filename, cam_img)


with open(r'./data/phco.txt','w') as f:
    for i,it in enumerate(re):
        f.write(str(i)+" "+str(int(it[0]))+" "+str(int(it[1])))
        f.write("\n")