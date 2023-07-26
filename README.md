# Correction of projector projection images for multi-projector fusion
## Introduction
This is an algorithm for image correction before fusion of multiple projectors, where the input image from each projector is corrected to stitch together to get a complete picture.

## Usage
We have realized the matching of the pixel coordinates of the projector image plane and the projected image through the medium of the camera, which is selected from FLIR.

1. Matching of projector image plane and projected image pixel coordinates
* You need to first to run `python capture_gray.py` get a set of Gray code photos.
* Execute `python test.py --mode matching`, it will help you to finish the decoding of Gray code, and after that, it will finish the matching of the pixel coordinates of the projected image plane and the projected image.
* The output contains two sets of correspondences for the x-axis and y-axis.

2. Correction of projected images
Image correction can be accomplished by running `python test.py --mode rendering` directly using the matching relation obtained in step 1.
