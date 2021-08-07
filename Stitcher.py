from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import vid_utils

#check all 4 videos have the same number of frames
#extract individual frames to stitch
#create new video?


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output image")
args = vars(ap.parse_args())

print('loading images...')
imagePaths = sorted(vid_utils.list_files('/Volumes/LexiZ/Research/SLEAP/PoseEstimation/Stitcher_Vid', 'avi'))
images = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

print('[INFO] stitching images...')
if imutils.is_cv3():
    stitcher = cv2.createStitcher()
else:
    cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)