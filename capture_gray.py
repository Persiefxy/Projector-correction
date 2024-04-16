import os
import cv2, numpy as np
import  screeninfo
import win32api
import win32con

#pattern_size = (1280, 720)
pattern_size = (640,360)
num_grids = (4, 4)
view_id = 0
pattern_dir = f'data/240415/patterns_2'
capture_dir = f'./data/240415/captured/position_{view_id:02d}a'
os.makedirs(pattern_dir, exist_ok=True)
os.makedirs(capture_dir, exist_ok=True)

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
#控制显示屏幕
def change_screen_resolution(width, height):
    # 获取当前显示器设备上下文
    device_context = win32api.EnumDisplayDevices()
    device_name = device_context.DeviceName

    # 新的分辨率设置
    mode = win32api.EnumDisplaySettings(device_name, win32con.ENUM_CURRENT_SETTINGS)
    mode.PelsWidth = width
    mode.PelsHeight = height

    # 改变分辨率
    win32api.ChangeDisplaySettings(mode, 0)


monitors = screeninfo.get_monitors()
change_screen_resolution(640,360 )
screen = monitors[-1]
width, height = screen.width, screen.height
# cv2.namedWindow("GrayCode", cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow("GrayCode", screen.x - 1, screen.y - 1)
# cv2.setWindowProperty("GrayCode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('GrayCode', grid_img)
#cv2.waitKey(0)

# system = PySpin.System.GetInstance()
# cam_list = system.GetCameras()
# print(cam_list)
# cam.Init()
#
# system = PySpin.System.GetInstance()
# cam_list = system.GetCameras()
# assert len(cam_list) > 0
# cam = cam_list[0]
# nodemap_tldevice = cam.GetTLDeviceNodeMap()
# node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
# assert PySpin.IsReadable(node_device_serial_number)
# cam.Init()
# nodemap = cam.GetNodeMap()
# sNodemap = cam.GetTLStreamNodeMap()
#
# # Change bufferhandling mode to NewestOnly
# node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
# assert PySpin.IsReadable(node_bufferhandling_mode) and PySpin.IsWritable(node_bufferhandling_mode)
# node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
# assert PySpin.IsReadable(node_newestonly)
# node_newestonly_mode = node_newestonly.GetValue()
# node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
#
# node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
# assert PySpin.IsReadable(node_acquisition_mode) and PySpin.IsWritable(node_acquisition_mode)
# node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
# assert PySpin.IsReadable(node_acquisition_mode_continuous)
# acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
# node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
#
# node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
# assert PySpin.IsReadable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto)
# node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
# assert PySpin.IsReadable(node_exposure_auto_off)
# exposure_auto_off = node_exposure_auto_off.GetValue()
# node_exposure_auto.SetIntValue(exposure_auto_off)
#
# node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
# assert PySpin.IsReadable(node_exposure_time) and PySpin.IsWritable(node_exposure_time)
# node_exposure_time.SetValue(120000.0)
#
# node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
# assert PySpin.IsReadable(node_gain_auto) and PySpin.IsWritable(node_gain_auto)
# node_gain_auto_off = node_gain_auto.GetEntryByName('Off')
# assert PySpin.IsReadable(node_gain_auto_off)
# gain_auto_off = node_gain_auto_off.GetValue()
# node_gain_auto.SetIntValue(gain_auto_off)
# node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
# assert PySpin.IsReadable(node_gain) and PySpin.IsWritable(node_gain)
# node_gain.SetValue(18.0)

#  Begin acquiring images
#cam.BeginAcquisition()
capture_mode = False


# 初始化相机
cam = cv2.VideoCapture(0)  # 假设0是相机的索引，根据实际情况调整
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
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
        cv2.waitKey(500)
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
        cv2.waitKey(500)

# 释放相机资源并关闭所有OpenCV窗口
cam.release()
cv2.destroyAllWindows()

# while True:
#     image_result = cam.GetNextImage(1000)
#     if image_result.IsIncomplete():
#         print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
#     else:
#         image_converted = image_result.Convert(PySpin.PixelFormat_BGR8)
#         cam_img = image_converted.GetNDArray()
#         preview = cv2.resize(cam_img, (0, 0), fx=0.5, fy=0.5)
#         image_result.Release()
#         cv2.imshow("Preview", preview)
#         key = cv2.waitKey(1) & 0xFF
#
#         # press space to save image
#         if key == 32:
#             i_code = 0
#             cv2.imshow('GrayCode', pattern_images[i_code])
#             cv2.waitKey(500)
#             capture_mode = True
#             continue
#         elif key == 27:
#             break
#         if capture_mode:
#             filename = f"{capture_dir}/gc_{i_code:04d}.png"
#             cv2.imwrite(filename, cam_img)
#             print(f"Saved {filename}")
#             i_code = i_code + 1
#             if i_code >= len(pattern_images):
#                 break
#             cv2.imshow('GrayCode', pattern_images[i_code])
#             cv2.waitKey(500)
#
# cv2.destroyAllWindows()
# cam.EndAcquisition()
# cam.DeInit()
# del cam
# cam_list.Clear()
# system.ReleaseInstance()

