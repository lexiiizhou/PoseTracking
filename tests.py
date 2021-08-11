import unittest
import SLEAP_batchTrack as sl
import os
import h5_to_csv
import vid_utils as vut
import subprocess

sl.vid_to_csv('/media/data/Sleap/probswitchVids/testVids', '/media/data/Sleap/models/TA210112_221325.single_instance.76')

# h5_to_csv.h5_to_csv('/media/data/Sleap/probswitchVids/testVids/A2A-19B_LT_20200728_LH_ProbSwitch_p159.avi'
#                     '.predictions.analysis.h5', 29)


# file_dir = '/media/data/Sleap/probswitchVids/testVids/A2A-19B_LT_20200728_LH_ProbSwitch_p159.avi.predictions.analysis.h5'
#
# session_name = file_dir.split('/')[-1].split('.')[0]
# print(os.path.dirname(os.path.realpath(file_dir)) + "/" + session_name)

