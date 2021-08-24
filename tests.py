import unittest
import SLEAP_batchTrack as sl
import os
import h5_to_csv
import vid_utils as vut
import subprocess
#
# sl.vid_to_csv('/media/data/Sleap/probswitchVids/batchTrainTest',
#               '/media/data/Sleap/models/210821_155119.single_instance.n=55')

vut.sort_files('/media/data/Sleap/probswitchVids/batchTrainTest')

# vut.sort_files('/media/data/Sleap/probswitchVids/testVids')

# print(os.path.isdir('/media/data/Sleap/probswitchVids/testVids/A2A-19B_LT_20200728_LH_ProbSwitch_p159'))
