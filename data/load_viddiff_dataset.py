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
from tqdm import tqdm
import logging
import hashlib


def load_viddiff_dataset(splits=["easy"], subset_mode="0"):
    """
    splits in ['ballsports', 'demo', 'easy', 'fitness', 'music', 'surgery']
    """
    dataset = load_dataset("viddiff/viddiff", cache_dir=None)
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
    # dataset = dataset.map(_clean_annotations)
    dataset = apply_subset_mode(dataset, subset_mode)

    return dataset


def load_all_videos(dataset, cache=True, do_tqdm=True):
    """ 
    Return a 2-element tuple. Each element is a list of length len(datset). 
    First list is video A for each datapoint as a dict with elements 
        path: original path to video 
        fps: frames per second 
        video: numpy array of the video shape (nframes,H,W,3)
    Second list is the same but for video B. 
    """

    all_videos = ([], [])
    # make iterator, with or without tqdm based on `do_tqdm`
    if do_tqdm:
        it = tqdm(dataset)
    else:
        it = dataset

    # load each video
    for row in it:
        videos = get_video_data(row['videos'], cache=cache)

        all_videos[0].append(videos[0])
        all_videos[1].append(videos[1])

    return all_videos


def _clean_annotations(example):
    # Not all differences in the taxonomy may have a label available, so filter them.

    differences_gt_labeled = {
        k: v
        for k, v in example['differences_gt'].items() if v is not None
    }
    differences_annotated = {
        k: v
        for k, v in example['differences_annotated'].items()
        if k in differences_gt_labeled.keys()
    }

    # Directly assign to the example without deepcopy
    example['differences_gt'] = differences_gt_labeled
    example['differences_annotated'] = differences_annotated

    return example


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
            dir_cache = Path("cache/cache_data")
            dir_cache.mkdir(exist_ok=True, parents=True)
            hash_key = get_hash_key(path + str(frames_trim))
            memmap_filename = dir_cache / f"memmap_{hash_key}.npy"

            if os.path.exists(memmap_filename):
                video_info = np.load(f"{memmap_filename}.info.npy",
                                     allow_pickle=True).item()
                video = np.memmap(memmap_filename,
                                  dtype=video_info['dtype'],
                                  mode='r',
                                  shape=video_info['shape'])
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
            np.save(f"{memmap_filename}.info.npy", {
                'shape': video.shape,
                'dtype': video.dtype
            })
            memmap = np.memmap(memmap_filename,
                               dtype=video.dtype,
                               mode='w+',
                               shape=video.shape)
            memmap[:] = video[:]
            memmap.flush()
            video = memmap

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


def _subsample_video(video: np.ndarray,
                     fps_original: int,
                     fps_target: int,
                     fps_warning: bool = True):
    """ 
    video: video as numby array (nframes, h, w, 3)
    fps_original: original fps of the video 
    fps_target: target fps to downscale to
    fps_warning: if True, then log warnings to logger if the target fps is 
        higher than original fps, or if the target fps isn't possible because 
        it isn't divisible by the original fps. 
    """
    subsample_time = fps_original / fps_target

    if subsample_time < 1 and fps_warning:
        logging.warning(f"Trying to subsample frames to fps {fps_target}, which "\
            "is higher than the fps of the original video which is "\
            "{video['fps']}. The video fps won't be changed for {video['path']}. "\
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


def get_hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_n_differences(dataset, config_n_differences: int | str | Path):
    """
    The maximum number of differences the model is allowed to make. 
    Either it's a single int, or its a path to a json `ndiff`, where n_differences
    is indexed by the data split and sample action, e.g.: 
        ndiff['fitness']['fitness_4'] = 8
    For split 'fitness' and action 'fitness_4'

    Returns: a list with length len(dataset), with an int for each sample. 
    """
    if type(config_n_differences) is int:
        n_differences = [config_n_differences] * len(dataset)
    else:
        path = Path(config_n_differences)
        if not path.exists():
            raise ValueError(
                f"Config value n_differences: [{n_differences}] must be an int " \
                "or a path to a json with per-action level stuff n_differences ")
        with open(path, 'r') as fp:
            lookup_ndiff = json.load(fp)
        n_differences = []
        for row in dataset:
            split = row['split']
            action = row['action']
            if split not in lookup_ndiff.keys(
            ) or action not in lookup_ndiff[split].keys():
                raise ValueError(
                    f"n_differences json at {path} has no entry for {(action, split)}"
                )
            n_differences.append(lookup_ndiff[split][action])

    return n_differences


if __name__ == "__main__":
    # dataset = load_viddiff_dataset(splits=['surgery','ballsports'])
    dataset = load_viddiff_dataset(splits=['demo'])
    videos = load_all_videos(dataset)
