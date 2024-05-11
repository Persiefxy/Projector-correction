from PIL import Image
import matplotlib.pyplot as plt
import cv2, numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box,MultiPolygon
from shapely.ops import unary_union
def dotmatrix(size,r=3,c=5):
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
        return result
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
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.show()
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
def check1map(i1,):
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
def check2map(i1,i2):
    plt.figure(figsize=(10, 5))

    # 复制并过滤nan
    ni1 = i1.copy()
    ni1[np.isnan(ni1)] = 0
    ni2 = i2.copy()
    ni2[np.isnan(ni2)] = 0

    # 第一个子图
    plt.subplot(1, 2, 1)
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

    # 第二个子图
    plt.subplot(1, 2, 2)
    plt.title('2')
    plt.imshow(i2)
    min_y, min_x = np.unravel_index(ni2.argmin(), ni2.shape)
    max_y, max_x = np.unravel_index(ni2.argmax(), ni2.shape)
    plt.plot(min_x, min_y, 'ro')
    plt.text(min_x, min_y, 'min:' + str(np.min(ni2)), fontsize=12)
    plt.plot(max_x, max_y, 'go')
    plt.text(max_x, max_y, 'max:' + str(np.max(ni2)), fontsize=12)
    plt.colorbar()
    plt.axis('equal')

    plt.legend()
def check2scatter(i1,i2,issub=False):
    # 计算全局最小值和最大值
    # global_min_val = min(np.min(ni1), np.min(ni2))
    global_min_val = -20

    global_max_val = max(np.max(i1), np.max(i2))+20

    # 设置X轴和Y轴的范围
    if issub:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('1')
        plt.scatter(i1[:, 0], i1[:, 1], s=1,c='red', label='1 Points')
        plt.xlim(global_min_val, global_max_val)  # X轴范围从0
        plt.ylim(global_min_val, global_max_val)  # X轴范围从0
        
        plt.legend()
        # plt.axis('equal')
        plt.subplot(1, 2, 2)
        plt.title('2')
        plt.scatter(i2[:, 0], i2[:, 1], s=1,c='blue', label='2 Points')
        plt.xlim(global_min_val, global_max_val)  # X轴范围从0
        plt.ylim(global_min_val, global_max_val)  # X轴范围从0
        plt.legend()
        # plt.show()
    else:
        plt.figure(figsize=(5, 5))
        plt.scatter(i1[:, 0], i1[:, 1], s=1,c='red', label='1 Points')
        plt.scatter(i2[:, 0], i2[:, 1], s=1,c='blue', label='2 Points')
        plt.xlim(global_min_val, global_max_val)  # X轴范围从0
        plt.ylim(global_min_val, global_max_val)  # X轴范围从0
        plt.legend()
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
    # 找到最大内接矩形
    max_rect = find_max_inner_rectangle(union_polygon)
    # 获取Box的边界坐标
    bounds = max_rect.bounds
    # 将边界坐标转换为NumPy数组
    return np.array(bounds)
if __name__ == "__main__":
    # re=resize_image("pic1920_1080.png",0.3)
    # re.save("pic0.3.png")
    image = cv2.imread("pic1920_1080.jpg")
    cropped_image = crop_image(image, (1280, 720))
    cv2.imwrite("pic1280_720.jpg", cropped_image)
