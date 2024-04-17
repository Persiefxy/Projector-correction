# Correction of projector projection images for multi-projector fusion

## Introduction

This is an algorithm for image correction before fusion of multiple projectors, where the input image from each projector is corrected to stitch together to get a complete picture.

## Usage

源数据输入来自data 主要是 capture 和phco.txt 和目标图像pic

数据生成到result 主要是 match.npy和data.txt 还有最终图片

```
python capture_gray.py
python test.py --mode matching
python test.py --mode matching --shadow_thresh 80 --code_thresh 40 --projector_id 0 --ph_coordinate './data/phco.txt' --gray_folder './data/240415/captured/position_00a/' --match_np "./result/match.npy"
python test.py --mode rendering
```


We have realized the matching of the pixel coordinates of the projector image plane and the projected image through the medium of the camera

1. Matching of projector image plane and projected image pixel coordinates
* You need to first to run `python capture_gray.py` get a set of Gray code photos.

* Execute `python test.py --mode matching`, it will help you to finish the decoding of Gray code, and after that, it will finish the matching of the pixel coordinates of the projected image plane and the projected image.

* The output contains two sets of correspondences for the x-axis and y-axis.
2. Correction of projected images
   Image correction can be accomplished by running `python test.py --mode rendering` directly using the matching relation obtained in step 1.

## Result

Here we show the fusion effect of the content projected by the two sets of projectors.

A projector
![avatar](/doc/projector_1.jpg)
B projector
![avatar](/doc/projector_2.jpg)
A&B projector
![avatar](/doc/projector_1&2.jpg)
Input image
![avatar](/pic.png)
Fusion results
![avatar](/doc/result.jpg)
