import unittest
import SLEAP_batchTrack as sl
import os
import vid_utils as vut
import subprocess

subprocess.run('sleap-track -h', shell=True)

# sl.vid_to_csv('/media/data/Sleap/probswitchVids/testVids', '/media/data/Sleap/models/TA210112_221325.single_instance.76')

# class TestBatchTrack(unittest.TestCase):
#
#     def test_list_files(self):
#         self.assertEqual(sl.list_files('/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid', 'avi'),
#                          ['/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample1_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi',
#                           '/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample2_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi',
#                           '/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample3_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi'],
#                          "wrong files")
#
#     def test_sleapTrack(self):
#         sl.vid_to_csv('/media/data/Sleap/probswitchVids/testVids', '/media/data/Sleap/models/TA210112_221325.single_instance.76')
#
# vut.sample_from_vid('/media/data/Sleap/probswitchVids/D1-27H_LT_20200625_RH_ProbSwitch_p206.avi',
#                     '/media/data/Sleap/probswitchVids/testVids', 900, 3)




