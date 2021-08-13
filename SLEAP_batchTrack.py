import os
import logging
import random
import h5_to_csv as hc
import vid_utils as vut
import subprocess


def write_log(name, log_out_folder):
    # create logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)

    # Assign a filehandler to logger instance
    fh = logging.FileHandler(str(log_out_folder) + '/' + str(name) + '.log')
    fh.setLevel(logging.ERROR)

    #format logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add filehandler to logging instance
    logger.addHandler(fh)

    return logger


def sort_files(dir):
    """
    Group video files, hfd5 files, and csv files into separate folders within the current directory
    :param dir: dir to be sorted
    :return:
    """


def sleaptrack(input_folder, model, logger):
    """
    Predict on all videos in input folder using the given model and output
    prediction as hdf5 files to input_folder
    :param input_folder: folder with all the avi files (can contain other types of files)
    :param logger: logger
    :param model: path to the model being used
    :return: sample framerate so that h5_to_csv can use it
    """

    avi = vut.list_files(input_folder, '.avi')
    sample_framerate, duration, frame_count = vut.get_duration(random.choice(avi))

    for vid in avi:
        vid_name = str(vid).split('/')[-1]
        if not os.path.isfile(str(vid) + '.predictions.slp'):
            try:
                print('predicting on ' + str(vid))
                subprocess.run('sleap-track ' + str(vid) + ' -m ' + str(model), shell=True)
                print('Finished predicting on ' + str(vid))
            except Exception as e:
                print('ERROR: ' + vid_name + ' see log file for detail')
                logger.error("something went wrong with" + vid_name + 'tracking: ' + str(e))
                continue
        try:
            print('converting predictions to hdf5 for ' + str(vid))
            subprocess.run('sleap-convert ' + str(vid) + '.predictions.slp ' + ' --format analysis' + ' -o ' +
                           str(vid) + '.predictions.analysis.h5', shell=True)
            print('Finished converting predictions to hdf5 for ' + str(vid))
        except Exception as e:
            print('ERROR: ' + vid_name + 'see log file for detail')
            logger.error("something went wrong with" + vid_name + 'file conversion: ' + str(e))
            continue
    return sample_framerate


def vid_to_csv(input_folder, model):
    """
    :param input_folder: Input folder that contains all video files
    :param model: trained model file
    :return: train, export hdf5, and perform kinematic analysis
    """
    logger_location = str(input_folder) + '/logger'
    if not os.path.isdir(str(input_folder) + '/logger'):
        os.mkdir(str(input_folder) + '/logger')
    logger = write_log(str(input_folder).split('/')[-1], logger_location)

    cmd = '. $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate sleap_env'
    subprocess.run(cmd, shell=True)
    framerate = sleaptrack(input_folder,  model, logger)
    h5 = vut.list_files(input_folder, '.analysis.h5')

    for i in h5:
        try:
            print("Generating Kinematics for " + i + '...')
            hc.h5_to_csv(i, framerate)
            print("Finished generating Kinematics for " + i + "!")
        except Exception as e:
            print('ERROR: ' + i + 'see log file for detail')
            logger.error('something went wrong with ' + i + "analysis: " + str(e))
            continue
