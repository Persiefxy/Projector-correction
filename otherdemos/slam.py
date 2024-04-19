import numpy as np
import matplotlib.pyplot as plt
def reconstruct_3d(gray_mapping, camera_intrinsic):
	#gray_mapping = np.hstack( (pro_know,cam_know)).tolist()

	#没有意义因为重建不出3d https://zhuanlan.zhihu.com/p/136827980
	#https://www.cnblogs.com/wangguchangqing/p/8335131.html
	#gray码+相机位姿可能能重建出3d  太复杂了
	# 张正友法中相机内参矩阵

	"""Logitech C920: fx=fy=912, cx=320, cy=240
	Microsoft LifeCam HD-3000: fx=fy=720, cx=320, cy=240
	Sony PlayStation Eye: fx=fy=640, cx=320, cy=240"""
	fx=fy=912
	cx=320
	cy=240
	camera_intrinsic = np.array([[fx, 0, cx],
								[0, fy, cy],
								[0, 0, 1]])

	# 初始化相机坐标结果
	cam_coords = []

	for [x, y, u, v] in gray_mapping:
		Xc = (u - cx) / fx * x
		Yc = (v - cy) / fy * y
		cam_coords.append([Xc, Yc])

	cam_coords = np.array(cam_coords)

	# 重建3D坐标
	Xc = cam_coords[:,0]
	Yc = cam_coords[:,1]
	Zc = np.ones_like(Xc) #修改Zc为和Xc一样的shape

	points_3d = np.vstack([Xc, Yc, Zc]).T

	print(points_3d)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.show()