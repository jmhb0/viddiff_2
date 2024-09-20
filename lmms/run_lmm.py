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

    # get dataset and videos
    dataset = lvd.load_viddiff_dataset([args.data.split],
                                       args.data.subset_mode)
    videos = lvd.load_all_videos(dataset, do_tqdm=True)

    # make prompts and call the lmm
    batch_prompts_text, batch_prompts_video = lu.make_text_prompts(
        dataset, videos, args.eval_mode, args.lmm)
    predictions = lu.run_lmm(batch_prompts_text,
                             batch_prompts_video,
                             args.lmm,
                             verbose=True)

    # do eval
    results = eval_viddiff.eval_viddiff(dataset,
                                        predictions_unmatched=predictions,
                                        eval_mode=args.eval_mode,
                                        seed=args.seed,
                                        n_differences=args.n_differences,
                                        results_dir=args.logging.results_dir)
    ipdb.set_trace()
    pass


if __name__ == "__main__":
    main()
