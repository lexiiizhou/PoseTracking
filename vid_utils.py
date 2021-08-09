import imageio
import os
import cv2


def path_prefix_free(path):
    symbol = os.path.sep
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol, 0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol) + len(symbol):]


def get_duration(filename):
    """
    :param filename: path to video file
    :return: fps in frames/sec, duration in seconds, and frame_count
    """
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    return fps, duration, frame_count


def chunk_video_sample_from_file(filename, out_folder, fps, duration):
    """
    Output a clip of a given length of video filename and output to out_folder
    :param filename:
    :param out_folder:
    :param fps: frames/sec
    :param duration: in seconds
    :return:
    """
    assert path_prefix_free(filename).count('.') == 1, 'has to contain only one .'
    file_code = path_prefix_free(filename).split('.')[0]
    suffix = path_prefix_free(filename).split('.')[1]
    w = imageio.get_writer(os.path.join(out_folder, f'{file_code}_sample.{suffix}'), format='FFMPEG',
                           mode='I', fps=fps) # figure out what mode means
    vid = imageio.get_reader(filename)
    for ith, img in enumerate(vid):
        if ith < duration * fps:
            print('writing', img.shape, ith)
            w.append_data(img)
        if ith >= duration * fps:
            print('all done')
            break
    w.close()


def sample_from_vid(vid_file, out_folder, sample_duration, n):
    """
    :param vid_file: file path to the video
    :param out_folder: save samples to
    :param sample_duration: length of sample in seconds
    :param n: sample size
    :return: save n samples of vid_file of length sample_duration to out_folder
    """

    def crop(start, end, input, output):
        str = "ffmpeg" + " -i " + input + " -ss  " + start + " -to " + end + " -c copy " + output
        os.system(str)

    startmin, startsec, starthr = 0, 0, 0
    start = str(format(starthr, '02d')) + ':' + str(format(startmin, '02d')) + ':' + str(format(startsec, '02d'))
    vid_duration = get_duration(vid_file)[1]
    assert n < vid_duration / sample_duration, 'video not long enough'
    vid_name = vid_file.split('/')[-1]

    i = 0
    while i < n:
        i += 1
        minute = format(int(sample_duration / 60) + int(startmin), '02d')
        seconds = format(int(sample_duration % 60) + int(startsec), '02d')
        hour = format(int(sample_duration / 360) + int(starthr), '02d')
        end = str(hour) + ':' + str(minute) + ':' + str(seconds)
        crop(start, end, vid_file, out_folder + "/" + "sample" + str(i) + '_' + vid_name)
        print('saved:' + out_folder + "/" + "sample" + str(i) + '_' + vid_name)
        print(end)
        startmin, startsec, starthr = minute, seconds, hour


def sample_using_timestamps(vid_file, out_folder, timestamps, sample_duration):
    """
    sample from video based on provided timestamps
    :param vid_file: file path to video
    :param out_folder: where to save samples
    :param timestamps: a list of timestamps
    :param sample_duration: duration of each sample centered around the timestamp
    :return:
    """


def list_files(dir, type):
    """
    List all files of a certain type in the given dir
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