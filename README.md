# Correction of projector projection images for multi-projector fusion
## Introduction
This is an algorithm for image correction before fusion of multiple projectors, where the input image from each projector is corrected to stitch together to get a complete picture.

## Usage

源数据输入来自data 主要是
数据生成到result 主要是 match.npy和data.txt 还有最终图片
python capture_gray.py
python test.py --mode matching
We have realized the matching of the pixel coordinates of the projector image plane and the projected image through the medium of the camera, which is selected from FLIR.

1. Matching of projector image plane and projected image pixel coordinates
* You need to first to run `python capture_gray.py` get a set of Gray code photos.
* Execute `python test.py --mode matching`, it will help you to finish the decoding of Gray code, and after that, it will finish the matching of the pixel coordinates of the projected image plane and the projected image.
* The output contains two sets of correspondences for the x-axis and y-axis.

2. Correction of projected images
Image correction can be accomplished by running `python test.py --mode rendering` directly using the matching relation obtained in step 1.

## Result

Here we show the fusion effect of the content projected by the two sets of projectors.

A projector
![avatar](/result/projector_1.jpg)
B projector
![avatar](/result/projector_2.jpg)
A&B projector
![avatar](/result/projector_1&2.jpg)
Input image
![avatar](/result/pic.png)
Fusion results
![avatar](/result/result.jpg)
