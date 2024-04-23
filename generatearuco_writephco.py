import os
import  screeninfo
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
arucox=aruco1.width
arucoy=aruco1.height
generated_aruco_img=aruco1.generate_aruco15_codes()
corners,ids,anchors=aruco1.generate_corners_anchors()
re=[]
for key, data in anchors.items():
    re.append(data)
width, height =  (1200, 600) 
def display1():
    #调用投影仪
    monitors = screeninfo.get_monitors()
    for i,monitor in enumerate(monitors):
        print(f"{monitor.name} - Resolution: {monitor.width}x{monitor.height} - ID: {i}" )
    screen = monitors[int(input("Enter monitor number: "))]
    # width, height = screen.width, screen.height
    
    cv2.namedWindow("Press r to shoot", cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow("Press r to shoot", screen.x - 1, screen.y - 1)
    cv2.setWindowProperty("Press r to shoot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Press r to shoot', generated_aruco_img)
    cv2.imwrite("./data/aruco15.png", generated_aruco_img)
    # while True:
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == ord('r'):
    #         print("拍摄")
    #         break
    cv2.waitKey(0)
    
def cam1():
    #调用摄像头
    cam = cs.Camera()
    cam.set_camera_settings((1280,720),-4)

    ret, cam_img = cam.read("preview")  # 从相机读取一帧
    if not ret:
        print('未能获取图像...')

    # preview = cv2.resize(cam_img, (0, 0), fx=0.5, fy=0.5)
    filename = "./data/aruco_new.png"
    cv2.imwrite(filename, cam_img)
    cv2.waitKey(0)
# display1()
# cam1()

with open(r'./data/phco.txt','w') as f:
    for i,it in enumerate(re):
        x=it[0]
        y=it[1]
        x=x*width/arucox
        y=y*height/arucoy
        f.write(str(i)+" "+str(int(x))+" "+str(int(y)))
        f.write("\n")