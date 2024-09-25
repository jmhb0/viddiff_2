"""
python scripts/save_videos.py
"""
import ipdb
import json
from typing import Literal
import warnings
import os
from datasets import load_dataset
from pathlib import Path
import logging
import sys
import numpy as np
import cv2
import imageio
import imageio_ffmpeg as ffmpeg
import tqdm
from PIL import Image, ImageFont, ImageDraw

logging.basicConfig(level=logging.INFO,
                    format='%(filename)s:%(levelname)s:%(message)s')

sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms import config_utils
from lmms import lmm_mcq_utils as lmu

results_dir = Path("scripts/results/dataset_properties")
results_dir.mkdir(exist_ok=True, parents=True)


def save_video(video_array, f_save, fps):
    assert video_array.ndim == 4 and video_array.shape[-1] == 3
    h, w = video_array.shape[1:3]
    
    writer = ffmpeg.write_frames(f_save, (w, h), fps=fps)
    writer.send(None)  # Initialize the writer
    
    for frame in video_array:
        writer.send(np.asarray(frame))
    
    writer.close()


def stack_videos(vid0, vid1, mode='h'):
    """
    vid0 and vid
    """
    max_frames = max(len(vid0), len(vid1))
    combined_frames = []

    for i in range(max_frames):
        frame0 = vid0[min(i, len(vid0) - 1)]
        frame1 = vid1[min(i, len(vid1) - 1)]

        # convert to PIL to stack since that's what `stack_images` uses
        frame0_pil = Image.fromarray(frame0)
        frame1_pil = Image.fromarray(frame1)
        combined_pil = stack_images(frame0_pil, frame1_pil, mode=mode)
        combined_np = np.array(combined_pil)

        combined_frames.append(combined_np)

    combined_video_array = np.array(combined_frames)

    return combined_video_array


def stack_images(img0: Image.Image,
                 img1: Image.Image,
                 mode: Literal['h', 'v'] = 'h',
                 resize_width: bool = False,
                 resize_height: bool = False):
    """ 
    Put a pair of pil images next to each other horizonally if mode='h' or 
    vertically if mode='v'.

    Another method `stack_images_seq` does the same thing but with a sequence 
    of images. 
    """
    # mode should be 'h' or 'v' for stack horizontally or vertically.
    # If resizing by width, adjust both images to the width of the smaller image
    if resize_width:
        min_width = min(img0.width, img1.width)
        img0 = img0.resize(
            (min_width, int(min_width * img0.height / img0.width)))
        img1 = img1.resize(
            (min_width, int(min_width * img1.height / img1.width)))

    # If resizing by height, adjust both images to the height of the smaller image
    if resize_height:
        min_height = min(img0.height, img1.height)
        img0 = img0.resize(
            (int(min_height * img0.width / img0.height), min_height))
        img1 = img1.resize(
            (int(min_height * img1.width / img1.height), min_height))

    # Stack images horizontally
    if mode == 'h':
        max_height = max(img0.height, img1.height)
        total_width = img0.width + img1.width
        dst = Image.new('RGB', (total_width, max_height))
        dst.paste(img0, (0, 0))
        dst.paste(img1, (img0.width, 0))

    # Stack images vertically
    elif mode == 'v':
        total_height = img0.height + img1.height
        max_width = max(img0.width, img1.width)
        dst = Image.new('RGB', (max_width, total_height))
        dst.paste(img0, (0, 0))
        dst.paste(img1, (0, img0.height))

    return dst


def save_video_pairs(split):
    dataset = lvd.load_viddiff_dataset([split], '0')
    videos = lvd.load_all_videos(dataset, do_tqdm=True)
    results_subdir = results_dir / f"video_samples_{split}"
    results_subdir.mkdir(exist_ok=True, parents=True)
    idx = 0
    for row, vid0, vid1 in tqdm.tqdm(zip(dataset, *videos),
                                     total=len(dataset)):
        # f0 = results_subdir / f"{row['sample_key']}_0.mp4"
        # f1 = results_subdir / f"{row['sample_key']}_1.mp4"
        # save_video(vid0['video'], f0, int(vid0['fps']))
        # save_video(vid1['video'], f1, int(vid1['fps']))

        vid_stacked = stack_videos(vid0['video'], vid1['video'], mode='h')
        f_save = results_subdir / f"sample_{row['sample_key']}_action_{row['action']}.mp4"
        save_video(vid_stacked, f_save, int(vid0['fps']))


def create_grid_video(rgb_video,
                      num_frames,
                      n_rows,
                      label_size=30,
                      pad_size=10):
    """
    How is this different from `create_image_grid_with_labels` below?
    Here you specify num_frames. ''
    """
    n, h, w, _ = rgb_video.shape
    n_cols = (num_frames + n_rows - 1) // n_rows

    # Create a blank canvas
    canvas_width = n_cols * (w + pad_size) + pad_size
    canvas_height = n_rows * (h + pad_size + label_size) + pad_size
    canvas = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))

    font = ImageFont.load_default(label_size)
    draw = ImageDraw.Draw(canvas)

    for i in range(num_frames):
        frame_idx = i * (n // num_frames)
        frame = Image.fromarray(rgb_video[frame_idx])
        col = i % n_cols
        row = i // n_cols
        x = col * (w + pad_size) + pad_size
        y = row * (h + pad_size + label_size) + pad_size

        canvas.paste(frame, (x, y + label_size))

        # Draw the label
        draw.text((x + w - label_size, y), str(i), fill="white", font=font)

    return canvas


if __name__ == "__main__":
    split = 'fitness'
    save_video_pairs(split)
