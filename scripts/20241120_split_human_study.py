"""
python -m ipdb scripts/20241120_split_human_study.py
"""

from pathlib import Path
import torch
import ipdb
import click
import json
import os
from datasets import load_dataset
import logging
import torch.nn.functional as F
import sys
import pandas as pd
import numpy as np
import cv2

sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms import config_utils
from lmms import lmm_utils as lu
import eval_viddiff

results_dir = Path("scripts/results/") / Path(__file__).stem
results_dir.mkdir(exist_ok=True, parents=True)

splits = ("diving", "fitness", "surgery", "ballsports", "music")
dataset = lvd.load_viddiff_dataset(splits=splits)
videos0, videos1 = lvd.load_all_videos(dataset, do_tqdm=True)

videos = videos0 + videos1
actions = [row['action'] for row in dataset] * 2
nframes = [len(v['video']) for v in videos]
fps = [v['fps'] for v in videos]
seconds = [n / f for (n, f) in zip(nframes, fps)]

df_stats = pd.DataFrame({"action": actions, "seconds": seconds})
action_to_seconds = df_stats.groupby(['action'])['seconds'].median().to_dict()
# I need the video length
actions = set(dataset['action'])
action_cnts = {}

def save_video_as_mp4(video_array, output_path, fps=30):
    num_frames, height, width, channels = video_array.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in video_array:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if frame_bgr.dtype != np.uint8:
            frame_bgr = (frame_bgr * 255).astype(np.uint8)
            
        out.write(frame_bgr)
    
    out.release()

import numpy as np
import imageio

def save_video_as_gif(video_array, output_path, fps=30, loop=0):
    # Validate input dimensions
    if video_array.ndim != 4 or video_array.shape[-1] != 3:
        raise ValueError("video_array must have shape (T, H, W, 3).")

    # Save as GIF with loop control
    with imageio.get_writer(output_path, mode='I', fps=fps, loop=loop) as writer:
        for frame in video_array:
            writer.append_data(frame)
    
    print(f"GIF saved to {output_path} with loop={loop}")
import numpy as np
from scipy.ndimage import zoom
def downscale_video(video_array, downscale_factor, antialias=True):
    """
    Downscale a video array using PyTorch with antialiasing.
    
    Parameters:
    video_array (numpy.ndarray): Array of shape (T, H, W, 3)
    downscale_factor (int or float): Factor by which to reduce height and width
    antialias (bool): Whether to apply antialiasing (default: True)
    
    Returns:
    numpy.ndarray: Downscaled video array in same dtype as input
    """
    # Record original dtype for later conversion
    orig_dtype = video_array.dtype
    need_uint8_convert = orig_dtype == np.uint8
    
    # Convert uint8 to float32
    if need_uint8_convert:
        video_array = video_array.astype(np.float32) / 255.0
    
    # Convert to torch tensor and move channel dimension to torch format
    # From (T, H, W, C) to (T, C, H, W)
    x = torch.from_numpy(video_array).permute(0, 3, 1, 2)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    
    # Calculate new size
    H, W = x.shape[2:]
    new_H = H // downscale_factor
    new_W = W // downscale_factor
    
    # Perform downscaling with antialiasing
    x_downscaled = F.interpolate(
        x,
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False,
        antialias=antialias
    )
    
    # Convert back to numpy array and restore channel order
    # From (T, C, H, W) to (T, H, W, C)
    result = x_downscaled.cpu().permute(0, 2, 3, 1).numpy()
    
    # Convert back to uint8 if necessary
    if need_uint8_convert:
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result

if 1:
    df = pd.DataFrame(dataset)
    cols = ['action','action_name', 'split_difficulty']
    df_show = df.groupby(cols).sample(1)[cols]
    df_show.to_csv(results_dir / "split_difficulty.csv")
    ipdb.set_trace()
ipdb.set_trace()
pass



# make the object
all_actions = {}  # across all actions
actions_recorded = set()  # just to keep track of which actions are included
videos_record = []
assert len(dataset) == len(videos0)
captions = []
str_log_all = []
for idx, row in enumerate(dataset):
    action = row['action']
    action_description = row['action_description']
    if action in actions_recorded:
        continue
    actions_recorded.add(action)
    videos_record += [videos0[idx]]

    # get the correct differences and sort correctly
    keys_gt = {k for k, v in row['differences_gt'].items() if v is not None}
    keys_gt = sorted(keys_gt, key=int)
    differences = {
        k: v
        for k, v in row['differences_annotated'].items() if k in keys_gt
    }

    def sort_dict(example_dict):
        return dict(sorted(example_dict.items(),
                           key=lambda item: int(item[0])))

    differences = sort_dict(differences)
    differences_for_prompt = {}

    # extra information
    for k, v in differences.items():
        diff_ = {}
        diff_['name'] = v['name']
        diff_['description'] = v['description']
        # diff_['num_frames'] = v['num_frames']
        diff_['description'] = v['description']

        differences_for_prompt[k] = diff_

    all_actions[action] = {
        "action_description": action_description,
        "differences": differences_for_prompt,
        "average_seconds_per_video": f"{action_to_seconds[action]:.1f}",
        # "split": "easy|medium|hard",
        # "split_reason": "...",
    }
    str_log = f"Code: {action}\n"
    str_log += f"Action: {action_description}\n\n"
    for k, v in differences_for_prompt.items():
        str_log += f"[{v['name']}] {v['description']}\n" 
    str_log_all.append(str_log)
    captions.append(json.dumps(all_actions[action], indent=4))


    ### now get a video for each action
    fps = videos_record[-1]['fps']
    video = videos_record[-1]['video']
    if video.shape[1] > 1000:
        downscale = 4
    else: 
        downscale = 2
    if video.shape[0] > 100:
        subsample = 4
        fps = fps // subsample
    else: 
        subsample = 1
    # save_video_as_mp4(video, "tmp.mp4", fps=fps)
    f_vid = results_dir / f"{action}.gif"

    save_video_as_gif(video[::subsample,::downscale, ::downscale], f_vid, fps=fps)
    # save_video_as_gif(downscale_video(video, downscale), f_vid, fps=fps)

    f_txt = results_dir / f"{action}.txt"
    with open(f_txt, "w") as fp: 
        fp.write(str_log)










