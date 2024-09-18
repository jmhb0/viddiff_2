import ipdb
import pdb
import os
import numpy as np
import json
import re
from PIL import Image
from pathlib import Path
from datasets import load_dataset
import decord
import lmdb
from tqdm import tqdm

import sys

sys.path.insert(0, ".")

from cache import cache_utils

cache_data = lmdb.open("cache/cache_data", map_size=int(1e12))


def load_viddiff_dataset(splits=["easy"], subset_mode="0"):
    dataset = load_dataset("viddiff/viddiff_0")
    dataset = dataset['test']

    def _filter_splits(example):
        return example["split"] in splits

    dataset = dataset.filter(_filter_splits)

    def _map_elements_to_json(example):
        example["videos"] = json.loads(example["videos"])
        example["differences_annotated"] = json.loads(
            example["differences_annotated"])
        example["differences_gt"] = json.loads(example["differences_gt"])
        return example

    dataset = dataset.map(_map_elements_to_json)
    dataset = apply_subset_mode(dataset, subset_mode)

    return dataset


def load_all_videos(dataset, do_tqdm=True):
    """ 
    Return a 2-element tuple. Each element is a list of length len(datset). 
    First list is video A for each datapoint as a dict with elements 
        path: original path to video 
        fps: frames per second 
        video: numpy array of the video shape (nframes,3,H,W)
    Second list is the same but for video B. 
    """

    all_videos = ([], [])
    # make iterator, with or without tqdm based on `do_tqdm`
    if do_tqdm:
        print(f"Loading videos")
        it = tqdm(dataset)
    else:
        it = dataset

    # load each video
    for row in it:
        videos = get_video_data(row['videos'], cache=True)
        all_videos[0].append(videos[0])
        all_videos[1].append(videos[1])
    return all_videos


def get_video_data(videos: dict, cache=True):
    """
    Pass in the videos dictionary from the dataset, like dataset[idx]['videos'].
    Load the 2 videos represented as numpy arrays. 
    By default, cache the arrays ... so the second time through, the dataset 
    loading will be faster. 

    returns: video0, video1
    """
    video_dicts = []

    for i in [0, 1]:
        path = videos[i]['path']
        assert Path(path).exists(
        ), f"Video not downloaded [{path}]\ncheck dataset README about downloading videos"
        frames_trim = slice(*videos[i]['frames_trim'])

        video_dict = videos[i].copy()

        if cache:
            hash_key = cache_utils.hash_key(path + str(frames_trim))
            video = cache_utils.get_from_cache_np(hash_key, cache_data)
            if video is not None:
                video_dict['video'] = video
                video_dicts.append(video_dict)
                continue

        is_dir = Path(path).is_dir()
        if is_dir:
            video = _load_video_from_directory_of_images(
                path, frames_trim=frames_trim)

        else:
            assert Path(path).suffix in (".mp4", ".mov")
            video, fps = _load_video(path, frames_trim=frames_trim)
            assert fps == videos[i]['fps']

        if cache:
            cache_utils.save_to_cache_np(hash_key, video, cache_data)

        video_dict['video'] = video
        video_dicts.append(video_dict)

    return video_dicts


def _load_video(f, return_fps=True, frames_trim: slice = None) -> np.ndarray:
    """ 
    mp4 video to frames numpy array shape (N,H,W,3).
    Do not use for long videos
    frames_trim: (s,e) is start and end int frames to include (warning, the range
    is inclusive, unlike in list indexing.)
    """
    vid = decord.VideoReader(str(f))
    fps = vid.get_avg_fps()

    if len(vid) > 50000:
        raise ValueError(
            "Video probably has too many frames to convert to a numpy")

    if frames_trim is None:
        frames_trim = slice(0, None, None)
    video_np = vid[frames_trim].asnumpy()

    if not return_fps:
        return video_np
    else:
        assert fps > 0
        return video_np, fps


def _load_video_from_directory_of_images(
    path_dir: str,
    frames_trim: slice = None,
    downsample_time: int = None,
) -> np.ndarray:
    """

    `path_dir` is a directory path with images that, when arranged in alphabetical
    order, make a video. 
    This function returns the a numpy array shape (N,H,W,3)  where N is the 
    number of frames. 
    """
    files = sorted(os.listdir(path_dir))

    if frames_trim is not None:
        files = files[frames_trim]

    if downsample_time is not None:
        files = files[::downsample_time]

    files = [f"{path_dir}/{f}" for f in files]
    images = [Image.open(f) for f in files]

    video_array = np.stack(images)

    return video_array


def _subsample_video(video: np.ndarray, fps_original: int, fps_target: int,
                     fps_warning: bool = True):
    """ 
    video: video as numby array (nframes, 3, h, w)
    fps_original: original fps of the video 
    fps_target: target fps to downscale to
    fps_warning: if True, then log warnings to logger if the target fps is 
        higher than original fps, or if the target fps isn't possible because 
        it isn't divisible by the original fps. 
    """
    subsample_time = fps_original / fps_target

    if subsample_time < 1 and fps_warning:
        logging.warning(f"Config lmm.video_representation='{video_representation}' " \
            f"for an fps of {fps_target}, which is higher than the fps of the original " \
            f"video which is {video['fps']}. The video fps won't be changed for {video['path']}. "\
            f"\nSupress this warning by setting config fps_warning=False")
        return video, fps_original, 1

    subsample_time_int = int(subsample_time)
    fps_new = int(fps_original / subsample_time_int)
    if fps_new != fps_target and fps_warning:
        logging.warning(f"Config lmm.fps='{fps_target}' but the original fps is {fps_original} " \
            f"so we downscale to fps {fps_new} instead. " \
            f"\nSupress this warning by setting config fps_warning=False")

    video = video[::subsample_time_int]

    return video, fps_new, subsample_time_int


def apply_subset_mode(dataset, subset_mode):
    """ 
    For example if subset_mode is "3_per_action" then just get the first 3 rows
    for each unique action. 
    Useful for working with subsets. 
    """
    match = re.match(r"(\d+)_per_action", subset_mode)
    if match:
        instances_per_action = int(match.group(1))
        action_counts = {}
        subset_indices = []

        for idx, example in enumerate(dataset):
            action = example['action']
            if action not in action_counts:
                action_counts[action] = 0

            if action_counts[action] < instances_per_action:
                subset_indices.append(idx)
                action_counts[action] += 1

        return dataset.select(subset_indices)
    else:
        return dataset


if __name__ == "__main__":
    # dataset = load_viddiff_dataset(splits=['surgery','ballsports'])
    dataset = load_viddiff_dataset(splits=['demo'])
    videos = load_all_videos(dataset)
