"""
python lmms/run_lmm_mcq.py
"""
import ipdb
import click
import json
import os
from datasets import load_dataset
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(filename)s:%(levelname)s:%(message)s')

sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms import config_utils
from lmms import lmm_utils as lu
import eval_viddiff


# yapf: disable
@click.command()
@click.option("--config", "-c", default="lmms/configs/base_lmm.yaml", help="config file")
@click.option("--name", "-n", default=None, help="experiment name which fixes the filename. Default value uses the config file value")
# yapf: enable
def main(config, name):
    # config
    args = config_utils.load_config(config)

    # get dataset, videos, and allowable n_differences
    split = 'easy'
    dataset = lvd.load_viddiff_dataset([split], args.data.subset_mode)
    videos = lvd.load_all_videos(dataset, do_tqdm=True)

    # make prompts and call the lmm
    batch_prompts_text, batch_prompts_video = lu.make_text_prompts(
        dataset,
        videos,
        n_differences=None,
        eval_mode=args.eval_mode,
        args_lmm=args.lmm)

    # double checking the videos make sense
    if 0:
        from scripts.save_videos import stack_videos, save_video, create_grid_video
        from PIL import Image
        import numpy as np

        if args.lmm.api == "openai":
            idx = 0
            # if args.lmm.api == "openai":
            #     vid0 = batch_prompts_video[idx][0]['video']
            vid_stacked = stack_videos(videos[idx][0]['video'],
                                       videos[idx][1]['video'],
                                       mode='h')
            save_video(vid_stacked, "tmp.mp4", videos[idx][0]['fps'])
            # first and last image
            Image.fromarray(batch_prompts_video[0][0]).save("tmp0.png")
            Image.fromarray(batch_prompts_video[0][-1]).save("tmpminus1.png")

            grid = create_grid_video(np.array(batch_prompts_video[idx]),
                                     num_frames=len(batch_prompts_video[idx]),
                                     n_rows=5)

    predictions = lu.run_lmm(batch_prompts_text,
                             batch_prompts_video,
                             args.lmm,
                             args.eval_mode,
                             n_differences=None,
                             verbose=True)
    metrics = eval_viddiff.eval_mcq_ab(dataset,
                                       predictions,
                                       results_dir=args.logging.results_dir)


if __name__ == "__main__":
    main()
