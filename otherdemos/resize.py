from PIL import Image
import matplotlib.pyplot as plt
import cv2, numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box,MultiPolygon
from shapely.ops import unary_union
# 创建一个大图，其中包含所有gray_col[index]的子图
def split_image(image):
    width, height = image.size
    left_half = image.crop((0, 0, width // 2, height))
    right_half = image.crop((width // 2, 0, width, height))
    return left_half, right_half
def checkacruco(corners,anchors,Aruco_img):
    fig, ax = plt.subplots()
    for anchor_id, center_point in anchors.items():
        plt.text(center_point[0], center_point[1], str(anchor_id), color='blue', fontsize=12)
    plt.imshow(Aruco_img, cmap='gray')
    plt.show()
def checkcol(rcol):
    n_subplots = rcol.shape[0]
    fig, axes = plt.subplots(1, n_subplots, figsize=(n_subplots * 5, 5))
    for i in range(n_subplots):
        axes[i].imshow(rcol[i], cmap='gray')
        axes[i].set_title(f'Gray Code Column {i}')
    plt.show()
def computeNumberOfPatternImages(width, height):
    assert width > 0 and height > 0
    n_cols = int(np.ceil(np.log2(width))) * 2 #ceil向上取整 宽度 log2
    n_rows = int(np.ceil(np.log2(height))) * 2
    return n_cols, n_rows
def findcontours(shadow_mask):
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 排序
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # Plotting contours using matplotlib
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=2)
    plt.title('Contours Plot')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
def dotmatrix(size,r=3,c=5):  #N,2的数组
        dx=size[0]/(c-1)
        dy=size[1]/(r-1)
        x=0
        y=0
        result=[]
        for i in range(r):
            for j in range(c):
                result.append([x,y])
                x=x+dx
            y=y+dy
            x =0
        return np.array(result)
def fourdotclock(size):  
    pts2 = np.float32([[0,0],[size[0],0],[size[0],size[1]],[0,size[1]]])
    return pts2
def fourdot(size):  
    pts2 = np.float32([[0,0],[size[0],0],[0,size[1]],[size[0],size[1]]])
    return pts2
def resize_image(image_path, ratio=0.2):
    with Image.open(image_path) as img:
        width, height = img.size
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_img = img.resize((new_width, new_height))
        return resized_img
def resize_img(img, ratio=0.2):
    original_width, original_height = img.shape[:2]
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_img = cv2.resize(img,( new_height,new_width), cv2.INTER_CUBIC)
    return resized_img
def resize_and_show(title,img, ratio=0.2):
    original_width, original_height = img.shape[:2]
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_img = cv2.resize(img,( new_height,new_width), cv2.INTER_CUBIC)
    cv2.imshow(title, resized_img)
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def plt_show(name,img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='equal')
    plt.title(name)
def plt_show_tuples(name,tupleslist):#元组的list
    # Convert list of tuples to separate lists of x and y coordinates
    x_coords, y_coords = zip(*tupleslist)
    # Create a scatter plot
    plt.figure(figsize=(8, 4))
    plt.scatter(x_coords, y_coords, color='green', label='Generated Points')
    plt.title(name)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
def plt_show_array(name,array):#必须是（xxx,2)的array
    # Plot the transformed points
    plt.figure(figsize=(8, 4))
    plt.scatter(array[:, 0], array[:, 1], color='purple', label='NDC Transformed Points')
    plt.title(name)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
def plt_show_withbox(img,array):
    imgwb1 = cv2.polylines(img, [np.int32(array)], True, (0, 255, 0), 1)
    plt.figure(figsize=(20, 20))
    plt_show("withbox",imgwb1)
def draw1map(points):
    """注意生成的图像会大于点10像素 输入N,2

    Args:
        i1 (_type_): _description_
    """

    max_x = int(np.max(points[:,0]))+10
    max_y = int(np.max(points[:,1]))+10
    # 创建一个新的空白图像。
    img = np.zeros((max_y,max_x,  3), dtype=np.uint8)
    # 绘制散点。
    for point in points:
        x = int(point[0])
        y = int(point[1])
        cv2.circle(img, (x, y), 2, (255, 255, 255), thickness=10)
    return img
def check1map(i1):
    plt.figure(figsize=(6, 6))
    # 复制并过滤nan
    ni1 = i1.copy()
    ni1[np.isnan(ni1)] = 0
    plt.title('1')
    plt.imshow(i1)
    min_y, min_x = np.unravel_index(ni1.argmin(), ni1.shape)
    max_y, max_x = np.unravel_index(ni1.argmax(), ni1.shape)
    plt.plot(min_x, min_y, 'ro')
    plt.text(min_x, min_y, 'min:' + str(np.min(ni1)), fontsize=12)
    plt.plot(max_x, max_y, 'go')
    plt.text(max_x, max_y, 'max:' + str(np.max(ni1)), fontsize=12)
    plt.colorbar()
    plt.axis('equal')
    return np.max(ni1)
def check2map(i1,i2):
    plt.figure(figsize=(10, 5))

    # 复制并过滤nan
    ni1 = i1.copy()

    ni2 = i2.copy()

    # 找到不包含NaN的最小值和最大值
    min_value = np.nanmin(ni1)
    max_value = np.nanmax(ni1)
    # 找到最小值和最大值所在的索引位置
    min_indices = np.where(ni1 == min_value)
    max_indices = np.where(ni1 == max_value)
    # 获取最小值和最大值的索引
    min_y, min_x = min_indices[0][0], min_indices[1][0]
    max_y, max_x = max_indices[0][0], max_indices[1][0]
    # 第一个子图
    plt.subplot(1, 2, 1)
    plt.title('1')
    plt.imshow(i1)

    plt.plot(min_x, min_y, 'ro')
    plt.text(min_x, min_y, 'min:' + str(min_value), fontsize=12)
    plt.plot(max_x, max_y, 'go')
    plt.text(max_x, max_y, 'max:' + str(max_value), fontsize=12)
    plt.colorbar()
    plt.axis('equal')
    # 找到不包含NaN的最小值和最大值
    min_value = np.nanmin(ni2)
    max_value = np.nanmax(ni2)
    # 找到最小值和最大值所在的索引位置
    min_indices = np.where(ni2 == min_value)
    max_indices = np.where(ni2 == max_value)
    # 获取最小值和最大值的索引
    min_y, min_x = min_indices[0][0], min_indices[1][0]
    max_y, max_x = max_indices[0][0], max_indices[1][0]


    # 第二个子图
    plt.subplot(1, 2, 2)
    plt.title('2')
    plt.imshow(i2)


    plt.plot(min_x, min_y, 'ro')
    plt.text(min_x, min_y, 'min:' + str(min_value), fontsize=12)
    plt.plot(max_x, max_y, 'go')
    plt.text(max_x, max_y, 'max:' + str(max_value), fontsize=12)
    plt.colorbar()
    plt.axis('equal')

    plt.legend()
def check2scatter(i1,i2,issub=False):
    # 计算全局最小值和最大值
    # global_min_val = min(np.min(ni1), np.min(ni2))
        # 复制并过滤nan
    ni1 = i1.copy()
    ni1[np.isnan(ni1)] = 0
    ni2 = i2.copy()
    ni2[np.isnan(ni2)] = 0
    global_min_val = -20

    global_max_val = max(np.max(ni1), np.max(ni2))+20

    # 设置X轴和Y轴的范围
    if issub:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].set_title('1')
        axs[0].scatter(i1[:, 0], i1[:, 1], s=1, c='red', label='pro Points')
        axs[0].set_xlim(global_min_val, global_max_val)
        axs[0].set_ylim(global_min_val, global_max_val)
        axs[0].legend()

        axs[1].set_title('2')
        axs[1].scatter(i2[:, 0], i2[:, 1], s=1, c='blue', label='cam Points')
        axs[1].set_xlim(global_min_val, global_max_val)
        axs[1].set_ylim(global_min_val, global_max_val)
        axs[1].invert_yaxis()  # 反转Y轴方向
        axs[1].legend()
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(i1[:, 0], i1[:, 1], s=1, c='red', label='1 Points')
        ax.scatter(i2[:, 0], i2[:, 1], s=1, c='blue', label='2 Points')
        ax.set_xlim(global_min_val, global_max_val)
        ax.set_ylim(global_min_val, global_max_val)
        ax.legend()
        ax.invert_yaxis()

        # plt.show()

def crop_image(image, crop_size,crop_point=(0,0)):
    """
    裁剪图片到指定区域。

    参数:
    image: 输入的图片。
    crop_area: 裁剪区域，格式为 ( width, height)。

    返回:
    裁剪后的图片。
    """
    x=crop_point[0]
    y=crop_point[1]
    width, height = crop_size
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image

def find_bounding_rectangle(points):
    # 合并所有点的坐标
    all_points = np.concatenate(points)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box
def find_enclosing_rectangle(points):
    # 合并所有点的坐标
    # 将顶点坐标转换为Shapely多边形对象
    polygon_list = [Polygon(points1) for points1 in points]

    union_polygon = polygon_list[0]
    i=1
    while i<len(polygon_list):
        union_polygon = union_polygon.union(polygon_list[i])
        i=i+1

    # 注意如果是分裂的区域就凉了   找到最大的并集区域
    # max_union = None
    # for union in unions:
    #     if max_union is None or union.area > max_union.area:
    #         max_union = union


    # 函数：找到给定多边形内的最大内接矩形（简化网格搜索方法）
    def find_max_inner_rectangle(polygon, step=5):
        bounds = polygon.bounds
        max_area = 0
        best_rect = None

        # 网格搜索
        for xmin in np.arange(bounds[0], bounds[2], step):
            for ymin in np.arange(bounds[1], bounds[3], step):
                for xmax in np.arange(xmin + step, bounds[2] + step, step):
                    for ymax in np.arange(ymin + step, bounds[3] + step, step):
                        candidate_rect = box(xmin, ymin, xmax, ymax)
                        if polygon.contains(candidate_rect):
                            area = candidate_rect.area
                            if area > max_area:
                                max_area = area
                                best_rect = candidate_rect
        return best_rect

if __name__ == "__main__":
    # re=resize_image("pic1920_1080.png",0.3)
    # re.save("pic0.3.png")
    image = cv2.imread("pic1920_1080.jpg")
    cropped_image = crop_image(image, (1280, 720))
    cv2.imwrite("pic1280_720.jpg", cropped_image)
    # 找到最大内接矩形
    # max_rect = find_max_inner_rectangle(union_polygon)
    # # 获取Box的边界坐标
    # bounds = max_rect.bounds
    # # 将边界坐标转换为NumPy数组
    # return np.array(bounds)