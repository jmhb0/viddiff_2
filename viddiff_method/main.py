import ipdb
import click
from typing import Dict, List, Tuple
import logging
from line_profiler import LineProfiler

import sys
sys.path.insert(0, "")

from proposer import Proposer
# from retriever import Retriever
# from frame_differencer import FrameDifferencer

from data import load_viddiff_dataset as lvd
from viddiff_method import config_utils
import eval_viddiff


logging.basicConfig(level=logging.INFO,
                    format='%(filename)s:%(levelname)s:%(message)s')


# yapf: disable
@click.command()
@click.option("--config", "-c", default="configs/base.yaml", help="config file")
@click.option("--name", "-n", default=None, help="experiment name which determines the filename. Default value uses the config file value")
@click.option("--seed", "-s", default=None, help="Random seed. Default value uses the config file value")
@click.option("--eval_mode", "-e", default=None, type=click.Choice([None,'0','1','2']), help="Eval mode. Default value uses the config file value")
# @click.option("--test_flip", "-f", default=None, help="Flip the order of videos to test sensitivity to order. Default value uses the config file value")
# @click.option("--subset_mode", "-s", default=None, help="Data subset mode (see configs/base.yaml). Default value uses the config file value")
# yapf enable
def main(config, name, seed, eval_mode):
    
    logging.info("Loading config")
    args = config_utils.load_config(config)
    
    logging.info(f"Loading dataset {args.data.split}")
    dataset = lvd.load_viddiff_dataset([args.data.split],
                                       args.data.subset_mode)
    # videos = lvd.load_all_videos(dataset, do_tqdm=True)

    logging.info(f"Running LLM proposer")
    proposer = Proposer(args.proposer, args.logging, dataset)
    proposals = proposer.generate_proposals()

    ipdb.set_trace()

    logging.info(f"Running frame retrieval")
    retriever = Retriever(args.retriever, args.logging, dataset, dataset_eval,
                          proposals)
    retrieved_frames = retriever.retrieve_frames()
    # ipdb.set_trace()

    logging.info(f"Running VLM frame differencing")

    frame_diferencer = FrameDifferencer(args.frame_differencer, args.logging,
                                        dataset, dataset_eval, proposals,
                                        retrieved_frames)
    sample_predictions = frame_diferencer.caption_differences()
    ipdb.set_trace()

    

if __name__ == "__main__":
    main()
