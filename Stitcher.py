import imutils
import tqdm
import os
from moviepy.editor import ImageSequenceClip
import numpy as np
import imutils
import cv2
import vid_utils

'''
General workflow
1. Compute SIFT key points and descriptors for images
2. Compute distances between every descriptor in one image and every descriptor in the other image
3. Select the top "m" matches for each descriptor of an image
4. Run RANSAC to estimate homography
5. Warp to align for stitching
6. Iterate through all frames in the vide and stitch using the homography generated (assuming 
    no shift in camera angle or zoom through out all frames)
'''

'''
How stitch multiple videos that don't just connect horizontally? 
List of key points and features variables? Or run the 2 image stitching script 3 times to 
stitch all 4 videos. This is kinda redundant maybe not. 

Mathier approach:
1. Compute all pairwise homographies. Given that we have 4, it'd be H01, H02, H03, H12, H13, H23)
2. Homography H01 warps image I_0 to I_1 and H02 warps I_0 to I_2 and so on
3. Select one anchor image which position will remain fixed. (H1 = anchor)
4. Find image that better align with anchor image based on maximum number of consistent matches 
    (e.g H3)
5. Update H3 = H1 * inverse(H13) = inverse(H13) = H31
6. Find image that better matches I1 or I3 (e.g. I2 matching I3)
7. Update H2 = H3 * H23
8. Repeat will all homographies for each image is updated. 
DO bundle adjustment to globally optimize alignment


less complicated but should work just fine:
panorama = image[0]
for i in 1:len(images)-1:
    panorama = stitch(panorama, image[i])
'''

# check that all 4 videos have the same number of frames
# if not, use timestamp to ensure all videos are the same length


class MultiVideoStitch:

    def __init__(self, video_path, num_of_vid, vid_out_path, video_out_width= 800, method=None):
        self.videos = video_path
        self.video_out_path = vid_out_path
        self.num_of_vid = num_of_vid
        self.video_out_width = video_out_width
        self.method = method

        # Homography Matrix
        self.homo_mat = None

    def detectAndExtract(self, image):
        """
        Extract key points and features of a given image
        :param image: Image
        :return: key points and features
        """
        assert self.method is not None, "Need to define a feature detection method"

        if self.method == 'sift':
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif self.method == 'surf':
            descriptor = cv2.xfeature2d.SURF_create()
        elif self.method == 'brisk':
            descriptor = cv2.xfeature2d.BRISK_create()
        elif self.method == 'orb':
            descriptor = cv2.ORB_create()

        (key_points, features) = descriptor.detectAndCompute(image, None)

        return key_points, features

    def generateMatcher(self, crossCheck):
        """
        :param crossCheck:
        :return: a Matcher object
        """
        if self.method == 'sift' or self.method == 'surf':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif self.method == 'orb' or self.method == 'brisk':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        return matcher

    def matchKeyPoints(self, featuresA, featuresB):
        matcher = self.generateMatcher(crossCheck=True)

        # Match descriptors
        best_matches = matcher.match(featuresA, featuresB)

        # Sort features in order of distance
        # Points with small distance (more similarity) are ordered first in the vector
        rawMatches = sorted(best_matches, key=lambda x: x.distance)
        print("Raw matches (Brute force):", len(rawMatches))
        return rawMatches

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch()

