import ipdb
from datasets import Dataset
from typing import Tuple
import numpy as np
import logging
from omegaconf.basecontainer import BaseContainer
from apis import openai_api

import sys
sys.path.insert(0, ".")

import data.load_viddiff_dataset as lvd
import lmms.lmm_prompts as lp

def make_text_prompts(dataset: Dataset, videos: Tuple, eval_mode: int,
                      args_lmm: BaseContainer):

    batch_prompts_text, patch_prompt_videos = [], []
    for i, row in enumerate(dataset):
        prompt_text, prompt_video = make_prompt(row['action_description'],
                                                videos[0][i], videos[1][i],
                                                eval_mode, args_lmm)
        batch_prompts_text.append(prompt_text)
        patch_prompt_videos.append(prompt_video)

    return batch_prompts_text, patch_prompt_videos


def make_prompt(action_description: str, video0: dict, video1: dict,
                eval_mode: int, args_lmm: BaseContainer):
    """
    create the text and video prompts 
    The possible representations are: 
        - frames 
    """
    # basic text prompting information
    if eval_mode == 0:
        prompt_text = lp.prompt_template_open
    elif eval_mode == 1:
        raise NotImplementedError()
    elif eval_mode == 2:
        raise NotImplementedError()
    else:
        raise ValueError()

    prompt_text = prompt_text.replace("{action_description}",
                                      action_description)
    prompt_text = prompt_text.replace("{n_differences}",
                                      str(args_lmm.n_differences))

    # handle the video representation
    if args_lmm.video_representation == "frames":

        # create the images prompt
        nframes = []
        fps_new_images = []
        prompt_videos = []
        assert type(args_lmm.fps) is int
        for video in (video0, video1):
            video['video'], fps_new, subsample_time_int = lvd._subsample_video(
                video['video'], video['fps_original'], args_lmm.fps,
                args_lmm.fps_warning)
            nframes.append(len(video['video']))
            fps_new_images.append(fps_new)
            prompt_videos += list(video['video'])

        assert fps_new_images[0] == fps_new_images[1]

        # describe the video representation
        video_rep_description = lp.video_rep_description_frames
        video_rep_description = video_rep_description.replace(
            "{vid0_nframes}", str(nframes[0]))
        video_rep_description = video_rep_description.replace(
            "{vid1_nframes}", str(nframes[1]))
        video_rep_description = video_rep_description.replace(
            "{fps}", str(fps_new_images[0]))

    else:
        raise ValueError(
            f"Config for lmm.video_representation [{video_representation}] not recognised"
        )

    prompt_text = prompt_text.replace("{video_representation_description}",
                                      video_rep_description)

    return prompt_text, prompt_videos


def run_lmm(batch_prompts_text, batch_prompts_video, args_lmm, verbose=True):
    """ 
    Assumes that the `batch_prompts_video` was formatted in an appropriate way
    for each api in args_lmm.api. For example, openai takes videos as sequeneces
    of images, so the `batch_prompts_video` is actually a list of images, and the
    text prompts in `batch_prompts_text` should explain that.
    """
    if args_lmm.api == "openai":
        assert args_lmm.video_representation == "frames"
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        ipdb.set_trace()
        if verbose:
            logging.info(
                f"Runnin model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        res = openai_api.call_gpt_batch(batch_prompts_text,
                                        batch_prompts_video,
                                        seeds=seeds,
                                        model='gpt-4o-mini')
        cost = sum([b[1] for b in res])
        logging.info(f"Cost for variation generation: ${cost:.4f}")
        predictions = [b[0]['differences'] for b in res]

    else:
        raise ValueError(
            f"Have not implemented baseline [{args_lmm.api}] in config")

    return predictions



