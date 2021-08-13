import unittest
import SLEAP_batchTrack as sl
import os
import h5_to_csv
import vid_utils as vut
import subprocess

# sl.vid_to_csv('/media/data/Sleap/probswitchVids/testVids', '/media/data/Sleap/models/TA210112_221325.single_instance.76')

# print(vut.get_duration('/media/data/Sleap/RRvids/testTmaze/sample2_testTmaze1.avi'))

# vut.sample_from_vid('/media/data/Sleap/RRvids/testTmaze/testTmaze1.avi',
#                     '/media/data/Sleap/RRvids/testTmaze', 180, 4)

vut.chunk_video_sample_from_file('/media/data/Sleap/RRvids/testTmaze/testTmaze1.avi',
                                 '/media/data/Sleap/RRvids/testTmaze', 30, 2039, 4549)
