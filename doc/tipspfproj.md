
demo.py 修改了 capturegray的filr相机问题
生成 data/patterns
照相出captured

python test.py --mode matching
args.gray_folder, args.ph_coordinate, parameters, pro_size, cam_size

   parameters = 

args.code_thresh, #将解码码的阈值 

args.shadow_thresh,阴影掩码的阈值 

args.projector_id]投影仪的ID 

三个参数组合成一个列表，以便后续使用



```markup-templating
 test 
+---> matching_test()
|         +----> load_images_from_folder()
|         |                        |
|         +----> gray_decode() <---+
|         |                        |
|         +----> Aruco_detect()    |
|         |                        |
|         +----> relation() 

|
+---> rendering_test() ----> cv2.remap()
```


D:\Downloads\BaiduSyncdisk\multi-projector-automatic-correction\my.m
