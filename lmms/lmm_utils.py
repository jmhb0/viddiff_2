import ipdb
from datasets import Dataset
import numpy as np
import logging
from omegaconf.basecontainer import BaseContainer
import numpy as np
import sys
import json
import pandas as pd
import re

sys.path.insert(0, ".")

from apis import openai_api
from apis import gemini_api
import data.load_viddiff_dataset as lvd
import lmms.lmm_prompts as lp


def make_text_prompts(dataset: Dataset, videos: tuple, n_differences: list,
                      eval_mode: int, args_lmm: BaseContainer):

    batch_prompts_text, patch_prompt_videos = [], []
    for i, row in enumerate(dataset):
        if eval_mode == 0:
            differences_annotated = None

        # collect the non-None annotated differneces and collect descriptions
        elif eval_mode == 1:
            keys_gt = {
                k
                for k, v in row['differences_gt'].items() if v is not None
            }
            differences_annotated = row['differences_annotated']
            differences_annotated = {
                k: v['description']
                for (k, v) in differences_annotated.items() if k in keys_gt
            }
            assert len(differences_annotated) == 1
            differences_annotated = list(differences_annotated.values())[0]

            n_differences = [None] * len(dataset)

        elif eval_mode == 2:
            keys_gt = {
                k
                for k, v in row['differences_gt'].items() if v in ('a', 'b')
            }
            differences_annotated = row['differences_annotated']
            differences_annotated = {
                k: v['description']
                for (k, v) in differences_annotated.items() if k in keys_gt
            }

            n_differences = [None] * len(dataset)

        else:
            raise ValueError()

        prompt_text, prompt_video = make_prompt(
            row['action_description'],
            videos[0][i],
            videos[1][i],
            eval_mode,
            args_lmm,
            n_differences[i],
            differences_annotated=differences_annotated, 
            model=args_lmm.model)
        batch_prompts_text.append(prompt_text)
        patch_prompt_videos.append(prompt_video)

    return batch_prompts_text, patch_prompt_videos


def make_prompt(action_description: str,
                video0: dict,
                video1: dict,
                eval_mode: int,
                args_lmm: BaseContainer,
                n_difference: int = None,
                differences_annotated: dict = None,
                model: str = None):
    """
    create the text and video prompts 
    The possible representations are: {'frames','video', 'first_frame'}
    """
    if eval_mode == 0:
        prompt_text = lp.prompt_template_open
        prompt_text = prompt_text.replace("{n_differences}", str(n_difference))
    elif eval_mode == 1:
        prompt_text = lp.prompt_template_mcq_ab
        prompt_text = prompt_text.replace(
            "{differences_annotated}",
            json.dumps(differences_annotated, indent=2))
    elif eval_mode == 2:
        prompt_text = lp.prompt_template_mode_2
        prompt_text = prompt_text.replace(
            "{differences_annotated}",
            json.dumps(differences_annotated, indent=2))
        if 'qwen' not in model.lower():
            target = {
                k: {
                    'description': v,
                    'prediction': "a|b"
                }
                for k, v in differences_annotated.items()
            }   
        # deal with this exception
        else:
            target = {
                k: {
                    'description': v,
                    'prediction': "..."
                }
                for k, v in differences_annotated.items()
            }   

        prompt_text = prompt_text.replace("{target_out}",
                                          json.dumps(target, indent=2))

    else:
        raise ValueError()

    prompt_text = prompt_text.replace("{action_description}",
                                      action_description)

    # all videos have tha subsampling step
    assert type(args_lmm.fps) is int
    # print("Before ", video0['video'].shape, video1['video'].shape)
    for video in (video0, video1):
        video['video'], fps_new, subsample_time_int = lvd._subsample_video(
            video['video'], video['fps_original'], args_lmm.fps,
            args_lmm.fps_warning)
    # print("After ", video0['video'].shape, video1['video'].shape)
    # print()

    # handle the video representation
    if args_lmm.video_representation == "frames":

        # create the images prompt
        nframes = []
        fps_new_images = []
        prompt_videos = []

        for video in (video0, video1):
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

    elif args_lmm.video_representation == "first_frame":
        video_rep_description = lp.video_rep_description_first_frame
        prompt_videos = [video0['video'][0], video1['video'][0]]

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
            eval_mode: int,
            n_differences: list[int],
            debug=None,
            verbose: bool = True):
    """ 
    Assumes that the `batch_prompts_video` was formatted in an appropriate way
    for each api in args_lmm.api. For example, openai takes videos as sequeneces
    of images, so the `batch_prompts_video` is actually a list of images, and the
    text prompts in `batch_prompts_text` should explain that.
    """
    if eval_mode in (0, 2):
        assert n_differences is not None
        json_mode = True
    else:
        assert n_differences is None
        json_mode = False

    if args_lmm.api == "openai":
        from apis import openai_api
        assert args_lmm.video_representation in ("frames", "first_frame")
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Running model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        res = openai_api.call_gpt_batch(batch_prompts_text,
                                        batch_prompts_video,
                                        seeds=seeds,
                                        model=args_lmm.model,
                                        debug=debug,
                                        json_mode=json_mode)
        cost = sum([b[1] for b in res])
        logging.info(f"Cost for lmm differences generation: ${cost:.4f}")
        predictions = [b[0] for b in res]

    elif args_lmm.api == "gemini":

        assert args_lmm.video_representation == "video"
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        res = gemini_api.call_gemini_batch(batch_prompts_text,
                                           batch_prompts_video,
                                           seeds=seeds,
                                           model=args_lmm.model,
                                           debug=debug,
                                           fps=args_lmm.fps_gemini)
        if eval_mode != 1:
            predictions = _reformat_malformed_json_prediction(
                [r for r in res[0]])
        else:
            predictions = [r for r in res[0]]

    elif args_lmm.api == "qwen":
        from apis import qwen_api

        assert args_lmm.video_representation == "video"
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Running model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )

        msgs, responses = qwen_api.call_qwen_batch(batch_prompts_text,
                                       batch_prompts_video,
                                       seeds=seeds,
                                       model=args_lmm.model,
                                       debug=debug,
                                       json_mode=json_mode)
        predictions = _reformat_malformed_json_prediction(msgs)

    else:
        raise ValueError(
            f"Have not implemented baseline [{args_lmm.api}] in config")

    if eval_mode == 0:
        predictions_final = _truncate_too_many_preds(predictions,
                                                     n_differences,
                                                     do_warning=True)
    elif eval_mode == 1:
        predictions_final = []
        pattern = r"answer is \(([ab])\)"
        for pred in predictions:
            match = re.search(pattern, pred)
            if match is not None:
                ans = match.group(1)
            else:
                ans = "-1"
            predictions_final.append(ans)

    elif eval_mode == 2:
        predictions_final = predictions

    return predictions_final


def _remove_trailing_commas_json(json_string):
    """ Some lmm outputs add a trailing string sometimes """
    # Remove trailing commas from objects
    json_string = re.sub(r',(\s*})', r'\1', json_string)

    # Remove trailing comma from the last object in the main object
    json_string = re.sub(r',(\s*})$', r'\1', json_string)

    return json_string


def _reformat_malformed_json_prediction(malformed_outputs, skip=False):

    # run the skip branch if we have high confidence the json will be correct
    if skip:
        predictions = []
        for pred in malformed_outputs:
            pred_dict = json.loads(_remove_trailing_commas_json(pred))
            predictions.append(pred_dict)
        return predictions

    prompts = [
        lp.prompt_reformat_malformed_json.replace("{llm_output}", g)
        for g in malformed_outputs
    ]
    seeds = [0] * len(prompts)
    res = openai_api.call_gpt_batch(prompts, seeds=seeds, model='gpt-4o-mini')
    predictions = [r[0] for r in res]

    return predictions


def _truncate_too_many_preds(predictions, n_differences: list[int],
                             do_warning: bool):
    """
    Just naiveley take the first `n_differences` values
    """
    for i, pred in enumerate(predictions):
        if len(pred) > n_differences[i]:

            if do_warning:
                logging.warning(f"Max {n_differences[i]} differences allowed, but "\
                    f"prediction {i} has {len(pred)}. Doing naive truncation.")

            predictions[i] = dict(list(pred.items())[:n_differences[i]])

    # double check that it worked
    assert all([
        len(pred) <= n_diff for pred, n_diff in zip(predictions, n_differences)
    ])

    return predictions
