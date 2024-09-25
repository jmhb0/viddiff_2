import ipdb
from datasets import Dataset
import numpy as np
import logging
from omegaconf.basecontainer import BaseContainer
import json
import pandas as pd

import data.load_viddiff_dataset as lvd
import lmms.lmm_prompts as lp


def make_text_prompts(dataset: Dataset, videos: tuple,
                      args_lmm: BaseContainer):

    batch_prompts_text, patch_prompt_videos = [], []
    for i, row in enumerate(dataset):
        keys_gt = {
            k
            for k, v in row['differences_gt'].items() if v is not None
        }
        differences_annotated = {
            k: v['description']
            for (k, v) in row['differences_annotated'].items() if k in keys_gt
        }

        prompt_text, prompt_video = make_prompt(row['action_description'],
                                                differences_annotated,
                                                videos[0][i], videos[1][i],
                                                args_lmm)
        batch_prompts_text.append(prompt_text)
        patch_prompt_videos.append(prompt_video)

    return batch_prompts_text, patch_prompt_videos


def make_prompt(action_description: str, differences_annotated: dict,
                video0: dict, video1: dict, args_lmm: BaseContainer):
    """
    create the text and video prompts 
    The possible representations are: 
        - frames 
    """
    prompt_text = lp.prompt_template_mcq_ab
    prompt_text = prompt_text.replace("{action_description}",
                                      action_description)
    prompt_text = prompt_text.replace("{differences_annotated}",
                                      json.dumps(differences_annotated))

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

        total_frames = nframes[0] + nframes[1]
        if total_frames > args_lmm.max_imgs:
            raise ValueError(f"Total frames [{total_frames}] is more than the "\
                "max frames set in the config lmms.max_imgs. Change the " \
                "max_frames or lower the config value for lmms.fps")

    elif args_lmm.video_representation == "video":
        video_rep_description = lp.video_rep_description_2_videos
        prompt_videos = [video0['video'], video1['video']]

    else:
        raise ValueError(
            f"Config for lmm.video_representation [{video_representation}] not recognised"
        )

    prompt_text = prompt_text.replace("{video_representation_description}",
                                      video_rep_description)

    return prompt_text, prompt_videos


def run_lmm(batch_prompts_text: list[str],
            batch_prompts_video: list[list[np.ndarray]],
            args_lmm: BaseContainer,
            verbose: bool = True):
    """ 
    Assumes that the `batch_prompts_video` was formatted in an appropriate way
    for each api in args_lmm.api. For example, openai takes videos as sequeneces
    of images, so the `batch_prompts_video` is actually a list of images, and the
    text prompts in `batch_prompts_text` should explain that.
    """
    if args_lmm.api == "openai":
        from apis import openai_api
        assert args_lmm.video_representation == "frames"
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Runnin model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        res = openai_api.call_gpt_batch(batch_prompts_text,
                                        batch_prompts_video,
                                        seeds=seeds,
                                        model=args_lmm.model)
        cost = sum([b[1] for b in res])
        logging.info(f"Cost for lmm differences generation: ${cost:.4f}")
        predictions = [b[0] for b in res]

    return predictions


def eval_mcq(dataset, predictions):
    results = []
    for i, pred in enumerate(predictions):
        row = dataset[i]
        sample_key = row['sample_key']
        differences_gt = row['differences_gt']
        differences_annotated = row['differences_annotated']

        for k, v in pred.items():
            res = [
                row['sample_key'], differences_gt[k], pred[k], row['action'],
                row['differences_annotated'][k]['name'],
                row['differences_annotated'][k]['description']
            ]
            results.append(res)

    df = pd.DataFrame(
        results,
        columns=['sample_key', 'gt', 'pred', 'action', 'name', 'description'])
    acc = (df['pred'] == df['gt']).sum() / len(df)
    print(acc)

    ipdb.set_trace()
    pass
