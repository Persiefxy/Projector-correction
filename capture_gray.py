import os
import cv2, numpy as np
import  screeninfo
import datetime
now = datetime.datetime.now()
date_str = now.strftime("%m%d")
time_str = now.strftime("%H%M")
import method.camera_screen as cs
cam_size = (1280, 720)
pattern_size = (1280,720) #(640,360)
num_grids = (4, 4)
view_id = 0
pattern_dir = f'data/{date_str}/patterns_2'
capture_dir = f'data/{date_str}{time_str}/captured/position_00/'
log_file = os.path.join(pattern_dir, 'log.txt')
if not os.path.exists(pattern_dir):
    os.makedirs(pattern_dir)
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)


def create_preview(pattern_size, grid_size, view_id):
    print(pattern_size)
    img = np.zeros((pattern_size[1] * grid_size[1], pattern_size[0] * grid_size[0], 3), np.uint8)
    img.fill(255)
    view_row = view_id // (pattern_size[0] - 1)
    view_col = view_id % (pattern_size[0] - 1)
    for i in range(pattern_size[1]):
        for j in range(pattern_size[0]):
            if (i+j) % 2 == 0:
                color = (0, 0, 0)
                if i >= view_row and i <= view_row + 1 and j >= view_col and j <= view_col + 1:
                    color = (0, 0, 255)
                img[i*grid_size[1]:(i+1)*grid_size[1], j*grid_size[0]:(j+1)*grid_size[0], :] = color
    return img

grayCode = cv2.structured_light.GrayCodePattern.create(pattern_size[0], pattern_size[1])
print(grayCode.getNumberOfPatternImages())
ret, pattern_images = grayCode.generate()
pattern_images = list(pattern_images)

pattern_with_border = []
for idx, img in enumerate(pattern_images):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 255, 255])
    pattern_with_border.append(img)

black_img = np.zeros((pattern_size[1], pattern_size[0]), dtype=np.uint8)
white_img = np.zeros((pattern_size[1], pattern_size[0]), dtype=np.uint8)
black_img, white_img = grayCode.getImagesForShadowMasks(black_img, white_img)
pattern_images.append(black_img)
pattern_images.append(white_img)

for i in (0, 1):
    for j in (0, 1):
        starry_map = np.zeros([2, 2], np.uint8)
        starry_map[i, j] = 255
        starry_map = np.tile(starry_map, (pattern_size[1] // 2, pattern_size[0] // 2))
        pattern_images.append(starry_map)

for idx, img in enumerate(pattern_images):
    cv2.imwrite(f'{pattern_dir}/pat_{idx:02d}.png', img)

print(f'Generated {len(pattern_images)} images')
grid_img = create_preview(num_grids, (pattern_size[0] // num_grids[0], pattern_size[1] // num_grids[1]), view_id)
#调用投影仪
monitors = screeninfo.get_monitors()
for i,monitor in enumerate(monitors):
    print(f"{monitor.name} - Resolution: {monitor.width}x{monitor.height} - ID: {i}" )
id=int(input("Enter monitor number: "))
screen = monitors[id]
black_screen = np.zeros((100, 100, 3), dtype='uint8')
black_screens = []
for i,monitor in enumerate(monitors):
    if i is not id:
        black_screens.append(monitor)
with open(log_file, 'w') as f:
    #print('拍摄文件夹%r'%capture_dir, file=f)
    print(f"屏幕：{screen.name} - Resolution: {screen.width}x{screen.height} - 时间: {time_str}", file=f)
    print(f"拍摄文件夹：{capture_dir}", file=f)
    print(f"cam大小：{cam_size} bytes", file=f)
    print(f"cam大小：{cam_size} bytes", file=f)
width, height = screen.width, screen.height
cv2.namedWindow("GrayCode", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("GrayCode", screen.x - 1, screen.y - 1)
cv2.setWindowProperty("GrayCode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('GrayCode', grid_img)
#cv2.waitKey(0)
capture_mode = False

# 初始化相机,cv2.CAP_DSHOW
cam = cv2.VideoCapture(0) # 假设0是相机的索引，根据实际情况调整
cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_size[1])
cam.set(cv2.CAP_PROP_EXPOSURE, -7)
cam.set(10, 100) # brightness     min: 0   , max: 255 , increment:1

# 主循环
while True:
    ret, cam_img = cam.read()  # 从相机读取一帧
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
        cv2.waitKey(800)
        capture_mode = True
        continue
    # 按ESC键退出
    elif key == 27:
        break
    if capture_mode:
        # 保存图像
        filename = f"{capture_dir}/gc_{i_code:04d}.png"
        cv2.imwrite(filename, cam_img)
        print(f"Saved {filename}")
        i_code += 1
        if i_code >= len(pattern_images):
            break
        cv2.imshow('GrayCode', pattern_images[i_code])
        cv2.waitKey(800)

# 释放相机资源并关闭所有OpenCV窗口
cam.release()
cv2.destroyAllWindows()


