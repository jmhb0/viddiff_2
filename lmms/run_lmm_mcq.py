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
from lmms import lmm_mcq_utils as lmu


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
    batch_prompts_text, batch_prompts_video = lmu.make_text_prompts(
        dataset, videos, args.lmm)
    predictions = lmu.run_lmm(batch_prompts_text,
                              batch_prompts_video,
                              args.lmm,
                              verbose=True)
    ipdb.set_trace()
    metrics = lmu.eval_mcq(dataset, predictions)



if __name__ == "__main__":
    main()
