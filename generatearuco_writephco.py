import os
import cv2
import numpy as np
import method.camera_screen as cs
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

def generate_aruco_image(  markersX = 5  ,         #X轴上标记的数量
            markersY = 3  ,          #EY轴上标记的数量   本例生成5x7的棋盘
            markerLength = 100, #标记的长度，单位是像素
            markerSeparation = 20, #每个标记之间的间隔，单位像素
            margins = 20 ,#标记与边界之间的间隔
            borderBits = 1, #标记的边界所占的bit位数
            showImage = True
            ):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins
    height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins


    anchors_ratios = {}#适配anchors
    i=0
    for y in range(markersY):
        for x in range(markersX):
            x1 = (x * (markerLength + markerSeparation) + margins) / width
            y1 = (y * (markerLength + markerSeparation) + margins) / height
            anchors_ratios.append({i: (x1, y1)})
            i=i+1

    myByteList = []
    for idx, code in enumerate(aruco_dict.bytesList):
        code = code[np.newaxis, :, :]
        bits = cv2.aruco.Dictionary_getBitsFromByteList(code, 4)
        bits = cv2.flip(bits, 1)
        code = cv2.aruco.Dictionary_getByteListFromBits(bits)
        myByteList.append(code[0])
    myByteList = np.stack(myByteList, axis=0)
    dict1 = cv2.aruco.Dictionary(myByteList, 4)
    board = cv2.aruco.GridBoard((markersX, markersY),float(markerLength),float(markerSeparation), dict1)
    img = cv2.aruco.drawPlanarBoard(board,(width,height),margins,borderBits=borderBits)
    return img, anchors_ratios


#re = copy((1199,599))
img,re=generate_aruco_image()
#调用投影仪
myscreen=cs.Screen()
monitors = myscreen.monitors
myscreen = monitors[myscreen.guiselect()]
width, height = myscreen.width, myscreen.height

cv2.namedWindow("ARTag", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("ARTag", myscreen.x - 1, myscreen.y - 1)
cv2.setWindowProperty("ARTag", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('ARTag', img)

#调用摄像头
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
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