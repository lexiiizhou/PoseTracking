import SLEAP_batchTrack as sl
import vid_utils as vt
import Stitcher


'''
FolderPaths is a list of folders that contains the videos you want to run. Note that these 
videos should be of the same type of animal (e.g. mouse) and should have similar lighting and zoom.
Otherwise the model used won't give accurate predictions. One example would be different mice 
performing the same behavior task. Each folder contains the behavioral recordings of each individual mouse. 
'''
folderPaths = ['/media/data/Sleap/probswitchVids/BatchTrain/D1-R35-RT']
modelPath = '/media/data/Sleap/models/210821_155119.single_instance.n=55'
sortFile = True


def run(folder_Path, model_Path, sort_File=False):
    sl.vid_to_csv(folder_Path, model_Path)
    if sort_File:
        vt.sort_files(folder_Path)


for i in folderPaths:
    run(i, modelPath, sortFile)
