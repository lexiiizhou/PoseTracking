import os
import logging
import random
import h5_to_csv as hc
import vid_utils as vut


def list_files(dir, type):
    """
    List all type files in dir
    :param dir: directory
    :param TYPE: str
    :return:
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(type):
                r.append(os.path.join(root, name))
    return r


def sort_files(folder):
    """
    Group video files, hfd5 files, and csv files into separate folders within the current directory
    :param folder:
    :return:
    """


def write_log(name, log_out_folder):
    # create logger instance
    logger = logging.getlogger(name)
    logger.setLevel(logging.ERROR)

    # Assign a filehandler to logger instance
    fh = logging.FileHandler(str(log_out_folder) + '/' + str(name) + '.txt')
    fh.setLevel(logging.ERROR)

    #format logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add filehandler to logging instance
    logger.addHandler(fh)

    return logger


def sleapTrack(input_folder, model, logger):
    """
    Predict on all videos in input folder using the given model and output
    prediction as hdf5 files to input_folder
    :param input_folder: folder with all the avi files (can contain other types of files)
    :param log_location: where to save exception logs
    :param model: path to the model being used
    :return: sample framerate so that h5_to_csv can use it
    """
    avi = list_files(input_folder, '.avi')
    sample_framerate, duration, frame_count = vut.get_duration(random.choice(avi))
    for vid in avi:
        vid_name = str(vid).split('/')[-1]
        vid_parent_dir = os.path.abspath(os.path.join(vid, os.pardir))
        try:
            os.system('sleap-track ' + str(vid) + ' -m ' + str(model))
        except:
            logger.exception("something went wrong with" + vid_name + 'tracking')
            continue
        try:
            os.system('sleap-convert ' + str(vid) + '.predictions.slp ' + ' --format h5 ' + ' -o ' + vid_parent_dir)
        except:
            logger.exception("something went wrong with" + vid_name + 'file conversion')
            continue
    return sample_framerate


def vid_to_csv(input_folder, log_location, model, output_folder, sort_files=False):
    """
    :param input_folder: Input folder that contains all video files
    :param log_location: Where to keep exception/error logs
    :param model: trained model file
    :return: train, export hdf5, and perform kinematic analysis
    """
    logger = write_log(str(input_folder).split('/')[-1], log_location)
    framerate = sleapTrack(input_folder,  model, logger)
    h5 = list_files(input_folder, '.analysis.h5')
    for i in h5:
        try:
            hc.h5_to_csv(i, framerate)
        except:
            logger.exception('something went wrong with' + i + "analysis")
            continue
    if sort_files is True:
        sort_files(input_folder)


