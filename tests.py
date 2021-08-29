import unittest
import SLEAP_batchTrack as sl
import os
import h5_to_csv
import vid_utils as vut
import subprocess
import Stitcher

stitch = Stitcher.MultiVideoStitch('/media/data/Sleap/RRvids/stitchertest', 4,
                                   '/media/data/Sleap/RRvids/stitchertest/stitched', display=True)
stitch.twoVideoStitch('/media/data/Sleap/RRvids/stitchertest/R1_cam_2021-04-30T16_48_39.avi',
                                         '/media/data/Sleap/RRvids/stitchertest/R2_cam_2021-04-30T16_48_39.avi')

# sl.vid_to_csv('/media/data/Sleap/probswitchVids/BatchTrain',
#               '/media/data/Sleap/models/210821_155119.single_instance.n=55')

# vut.sort_files('/media/data/Sleap/probswitchVids/batchTrainTest')

# vut.sort_files('/media/data/Sleap/probswitchVids/testVids')

# print(os.path.isdir('/media/data/Sleap/probswitchVids/testVids/A2A-19B_LT_20200728_LH_ProbSwitch_p159'))


