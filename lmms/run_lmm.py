import ipdb
import click

import json
import os
from datasets import load_dataset

import sys 
sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms.configs import config_utils


@click.command()
@click.option("--config", "-c", default="lmms/configs/base_lmm.yaml", help="config file")
@click.option("--name", "-n", default=None, 
    help="experiment name which fixes the filename. Default value uses the config file value")


def main(config, name):
    args = config_utils.load_config(config)
    
    # dataset and videos
    dataset = lvd.load_viddiff_dataset([args.data.split])
    dataset = lvd.apply_subset_mode(dataset, args.data.subset_mode)
    videos = lvd.load_all_videos(dataset, do_tqdm=True)

    ipdb.set_trace()
    pass

    
if __name__ == "__main__":
    main()




