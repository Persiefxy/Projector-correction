from scipy.interpolate import griddata
import math
import cv2, numpy as np

# The matching of each point in the image plane of the projector and the camera is accomplished using interpolation, and unmatched points are recorded as Nan
def pro_cam_match(cmr_match_pjt, cam_size):
    pro_know = [] # The coordinates of the projection plane
    cam_know = [] # The coordinates of the camera plane
    for i in cmr_match_pjt:
        pro_know.append(i[1:])
        cam_know.append(cmr_match_pjt[i][0])
    pro_know = np.array(pro_know)
    print(pro_know.shape)
    cam_know = np.array(np.array(cam_know).reshape(-1,2))
    # Interpolation to achieve a one-to-one correspondence between the projection plane and the camera plane results are stored in cam_pro
    # (how the points in pro are filled into cam)
    grid_x = np.linspace(0, cam_size[0]-1, cam_size[0])  # x coordinate range
    grid_y = np.linspace(0, cam_size[1]-1, cam_size[1])  # y coordinate range
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    # print('grid_x is :',grid_x)
    # print('grid_y is :',grid_y)
    points_to_remap = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    cam_pro = griddata(cam_know, pro_know, points_to_remap, method='cubic')
    #cam_pro = griddata(cam_know, pro_know, (grid_x,grid_y), method='cubic')
    print(cam_pro)
    print(cam_pro.shape)
    map_x = np.zeros([cam_size[1], cam_size[0]], dtype=float)
    map_y = np.zeros([cam_size[1], cam_size[0]], dtype=float)
    for idx, num in enumerate(cam_pro):
        map_x[idx // cam_size[0]][idx % cam_size[0]] = num[0]
        map_y[idx // cam_size[0]][idx % cam_size[0]] = num[1]
    map_x, map_y = map_x.reshape(-1, map_x.shape[0], map_x.shape[1]), map_y.reshape(-1, map_x.shape[0], map_x.shape[1])
    map_matrix = np.concatenate((map_x, map_y), axis=0)
    return map_matrix

# Matching of the projector image plane and the calibrated points in the projected image
def pro_real_match(transformational_matrix, anchors, ph_coordinate):
    cam_know = [] # The position of the Aruco code in the image plane of the camera
    idx_Aruco = [] # Aruco code number
    for key, value in anchors.items():
        cam_know.append(np.array([int(value[1]), int(value[0])]))
        idx_Aruco.append(key)
    cam_know = np.array(cam_know)
    # Matching of projector image plane and projected image
    pro_real = {} # The coordinates of the pointscorresponding to the pixel points in the projector.
    for n, m in enumerate(cam_know):
        pro_real[idx_Aruco[n]] = np.array([(transformational_matrix[0][m[0]][m[1]]), (transformational_matrix[1][m[0]][m[1]])])
    # Reading calibrated projected image coordinates
    real_dic = {} # Aruco code in the coordinates of the projected image and its corresponding id
    real_np = np.loadtxt(ph_coordinate, encoding='utf-8', dtype=float)
    for i in real_np:
        real_dic[i[0]] = [i[1], i[2]]
    pro = [] # Pixel coordinates of points in the projector that match the projected image
    real = [] # Pixel coordinates of points where the projected image matches the projector
    # 记录投影平面和投影图像的匹配关系
    for key, value in pro_real.items():
        pro.append(value)
        real.append(real_dic[key])
    pro = np.array(pro)
    real = np.array(real)
    return pro, real

# Predicting the match between the projector image plane and the failure of the calibration points to match in the projected image
def predict_unknow(pro, real):
    pro_unknow = [] # Unknown point in projected coordinates
    real_unknow = [] # pro_unknow corresponds to the points in the projected image
    pro_know = [] # Projected coordinates of known points
    real_know = [] # pro_know corresponds to the point of the projected image
    print('pro is :',pro)
    for idx, num in enumerate(pro):
        if(math.isnan(num[0])):
            pro_unknow.append(num)
            real_unknow.append(real[idx])
        else:
            pro_know.append(num)
            real_know.append(real[idx])
    # Predicting the value of an unknown point by transformation
    if (len(real_unknow)!=0):
        pro_unknow = np.array(pro_unknow)
        real_unknow = np.array(real_unknow)
        pro_know = np.array(pro_know)
        real_know = np.array(real_know)
        print(real_know)
        print('the shape of real_know:',real_know.shape)
        print('the shape of pro_know:',pro_know.shape)
        M, _ = cv2.findHomography(real_know, pro_know) # transformation matrix
        pro_new = cv2.perspectiveTransform(real_unknow.reshape(-1, 1, 2), M).reshape(-1,2) # Predicted results
        # Record of the matching relationship between the calibration point (real_all) and the projector image plane (pro_all) point in projected images
        pro_all = np.vstack((pro_know,pro_new))
        real_all =  np.vstack((real_know,real_unknow))
    else:
        pro_all = np.array(pro_know)
        real_all =  np.array(real_know)
    return pro_all, real_all

# One-to-one matching of projector image plane coordinates and projected image pixel coordinates
def get_transformational_matrix(pro_all, real_all, pro_size):
    # Determine how points in the image plane of the projector are projected into the projected image by interpolation
    grid_x = np.linspace(0, pro_size[0]-1, pro_size[0])
    grid_y = np.linspace(0, pro_size[1]-1, pro_size[1])
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    points_to_remap = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    remapped_points = griddata(pro_all,real_all, points_to_remap, method='cubic')
    mapx = np.zeros([pro_size[1], pro_size[0]], dtype=float)
    mapy = np.zeros([pro_size[1], pro_size[0]], dtype=float)
    for idx, num in enumerate(remapped_points):
        mapx[idx // pro_size[0]][idx % pro_size[0]] = num[0]
        mapy[idx // pro_size[0]][idx % pro_size[0]] = num[1]
    mapx = np.float32(mapx)
    mapy = np.float32(mapy)
    return mapx, mapy

# Establishing a match between projector image plane coordinates and projected image pixel coordinates
def relation(anchors, cmr_match_pjt, ph_coordinate, pro_size, cam_size):
    transformational_matrix = pro_cam_match(cmr_match_pjt, cam_size)
    pro, real = pro_real_match(transformational_matrix, anchors, ph_coordinate)
    pro_all, real_all = predict_unknow(pro, real)
    mapx, mapy = get_transformational_matrix(pro_all, real_all, pro_size)
    return mapx, mapy