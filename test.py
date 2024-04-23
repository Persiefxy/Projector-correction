import argparse
import cv2, numpy as np
import os,screeninfo
from method.decode_gray import gray_decode
from method.match import relation
import method.camera_screen as cs
import datetime
now = datetime.datetime.now()
date_str = now.strftime("%m%d")
time_str = now.strftime("%H%M")

# Aruco code position detection in the camera image plane
def Aruco_detect(gray):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    myByteList = []
    for idx, code in enumerate(aruco_dict.bytesList):
        code = code[np.newaxis, :, :]
        bits = cv2.aruco.Dictionary_getBitsFromByteList(code, 4)
        bits = cv2.flip(bits, 1)
        code = cv2.aruco.Dictionary_getByteListFromBits(bits)
        myByteList.append(code[0])
    myByteList = np.stack(myByteList, axis=0)
    dict1 = cv2.aruco.Dictionary(myByteList, 4)
    parameters =  cv2.aruco.DetectorParameters()
    result = {}
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dict1, parameters=parameters)
    #print('corners: ', len(corners))
    #corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #assert (len(corners)!=0), print("Please retest")
    print('corners :',len(corners))
    if len(corners) == 0:
        print("Please retest")
        raise AssertionError
    if ids is not None:
        corners_1 = np.array(corners).reshape(-1, 4, 2)
        center_points = np.mean(corners_1, 1)
    for i, idx in enumerate(ids):
        idx = int(idx)
        result[idx] = center_points[i]

    print('result is :',result)
    return result
def myanchors(fourpointpath=""):#定义一个函数返回aruco标记中心点
    aruco=[]
    real_np = np.loadtxt(fourpointpath, encoding='utf-8', dtype=float)
    for i in real_np:
        aruco[i[0]] = [i[1], i[2]]
    return aruco
# Load captured Gray code image for decoding
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Complete matching of projected image pixel coordinates to projector pixel coordinates
def matching_test(images_folder, arucodir,ph_coordinate, parameters, pro_size, cam_size,type="aruco"):
    images_list = load_images_from_folder(images_folder)
    cmr_match_pjt = gray_decode(images_list,pro_size,parameters)
    img = cv2.imread(arucodir)
    if type == "aruco":
        anchors = Aruco_detect(img)
    else:
        anchors=myanchors()
    map_x, map_y = relation(anchors, cmr_match_pjt, ph_coordinate, pro_size, cam_size)
    map_x, map_y = map_x.reshape(1, -1, map_x.shape[0], map_x.shape[1]), map_y.reshape(1, -1, map_x.shape[0], map_x.shape[1])
    map_matrix = np.concatenate((map_x, map_y), axis=1)
    return map_matrix

# Corrects the projected image according to the matching result
def rendering_test(image, map_matrixs, output_dir):
    for idx,map_matrix in enumerate(map_matrixs):
        part = cv2.remap(image, map_matrix[0], map_matrix[1], interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f'{output_dir}/{idx}.png', part)
    #调用投影仪
    monitors = screeninfo.get_monitors()
    for i,monitor in enumerate(monitors):
        print(f"{monitor.name} - Resolution: {monitor.width}x{monitor.height} - ID: {i}" )
    screen = monitors[int(input("Enter monitor number: "))]
    width, height = screen.width, screen.height
    cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow("result", screen.x - 1, screen.y - 1)
    cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img = cv2.imread(f'{output_dir}/0.png')
    cv2.imshow('result', img)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord(' '):
    #         break
    cv2.waitKey(0)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pro_size = (1280,852)  # Projector image plane size 2560*640/3840)
    
    
    cam_size  =(1280, 720) # Camera image plane size
    arucodir='C:/Users/robert/source/repos/3BPM/test3\WindowsFormsApp1/bin/Debug/net6.0-windows10.0.17763.0/pic2.png'
    parser = argparse.ArgumentParser(
    description='Projector correction')
    parser.add_argument('--shadow_thresh', type=int, default=80,
                        help='The threshold of the shadow mask')
    parser.add_argument('--code_thresh', type=int, default=40,
                        help='The threshold of the decoded code')
    parser.add_argument('--projector_id', type=int, default = 0,
                        help='The id of the projector')
    parser.add_argument('--mode',type=str, default = "rendering",
                        help='1. matching, 2. rendering')
    parser.add_argument('--ph_coordinate', type=str, default = './data/phco.txt',
                        help='Projection image coordinates, eg: "./ph_coordinate.txt"')
    parser.add_argument('--gray_folder', type=str,default=r"data\04232022\captured\position_00"
                        #default = f'./data/{date_str}/captured/position_{time_str}/'
                        ,help='The folder where Gray codes are stored')
    parser.add_argument('--match_np', type=str, default = "./result/match.npy",
                        help='The file name where the matching results are stored, eg: "./match.npy"')
    parser.add_argument('--test_image', type=str, default = "./picc.png",
                        help='Test image, eg: "./test.png"')
    parser.add_argument('--output_dir', type=str, default = "./result/",
                        help='output image folder, eg: "./result/"')
    args = parser.parse_args()

    parameters = [args.code_thresh, args.shadow_thresh, args.projector_id]

    os.makedirs("./result/", exist_ok=True)
    if(args.mode == 'matching'):
        map_matrix = matching_test(args.gray_folder, arucodir,args.ph_coordinate, parameters, pro_size, cam_size)
        np.save(args.match_np, map_matrix)

    if(args.mode == 'rendering'):
        image = cv2.imread(args.test_image)
        matricies = np.load(args.match_np)
        rendering_test(image, matricies, args.output_dir)

