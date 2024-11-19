import torch
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
    dataset = lvd.load_viddiff_dataset([args.data.split],
                                       args.data.subset_mode)
    videos = lvd.load_all_videos(dataset, do_tqdm=True)
    n_differences = lvd.get_n_differences(dataset, args.lmm.n_differences)
    dataset, videos = _filter_bad_vids(dataset, videos, args.lmm.api)

    # make prompts and call the lmm
    batch_prompts_text, batch_prompts_video = lu.make_text_prompts(
        dataset, videos, n_differences, args.eval_mode, args.lmm)
    # debug = dataset['sample_key']
    predictions = lu.run_lmm(batch_prompts_text,
                             batch_prompts_video,
                             args.lmm,
                             args.eval_mode,
                             n_differences,
                             # debug=debug,
                             verbose=True)

    # do eval
    metrics = eval_viddiff.eval_viddiff(dataset=dataset,
                                        predictions_unmatched=predictions,
                                        eval_mode=args.eval_mode,
                                        n_differences=n_differences,
                                        seed=args.seed,
                                        results_dir=args.logging.results_dir)
    print(metrics)

    
    
def _filter_bad_vids(dataset, videos, api):
    bad_keys = []
    if api == 'openai':
        pass
        # bad_keys += ["surgery_76","surgery_74","surgery_94"]
        # , "surgery_113", "surgery_114", "surgery_115", "surgery_116", "surgery_117", "surgery_118"]
        # bad_keys += [f'surgery_{i}' for i in range(113,170)]
    elif api=="gemini":
        pass
       # bad_keys += ["surgery_115", "surgery_131", "surgery_138", "surgery_151", "surgery_165"]
       # bad_keys += ["ballsports_5", "ballsports_11","ballsports_14", "ballsports_16", "ballsports_18"] 
       # bad_keys += ["ballsports_0", "ballsports_1","ballsports_2", "ballsports_3", "ballsports_4"] 

    bad_idxs = [i for i, s in enumerate(dataset['sample_key']) if s in bad_keys]

    dataset_filtered = dataset.filter(lambda _, idx: idx not in bad_idxs, with_indices=True)
    
    videos_filtered = [[],[]] 

    videos_filtered[0] = [v for i, v in enumerate(videos[0]) if i not in bad_idxs]
    videos_filtered[1] = [v for i, v in enumerate(videos[1]) if i not in bad_idxs]
    assert len(dataset_filtered) == len(videos_filtered[0])
    
    return dataset_filtered, videos_filtered



if __name__ == "__main__":
    main()
