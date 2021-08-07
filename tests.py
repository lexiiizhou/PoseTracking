import unittest
import SLEAP_batchTrack as sl
import os, sys


class TestBatchTrack(unittest.TestCase):

    def test_list_files(self):
        self.assertEqual(sl.list_files('/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid', 'avi'),
                         ['/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample1_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi',
                          '/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample2_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi',
                          '/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample3_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi'],
                         "wrong files")

    def test_batchTrack(self):
        print('sleap-track /Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid/sample1_D1-27H_LT_20200625_RH_ProbSwitch_p206.avi -m /Volumes/LexiZ/Research/SLEAP/PoseEstimation/dreaddRotation')
        # sl.sleapTrack('/Volumes/LexiZ/Research/SLEAP/PoseEstimation/TestVid',
        #               '/Volumes/LexiZ/Research/SLEAP/PoseEstimation/dreaddRotation')

        os.system('python3 --version')




