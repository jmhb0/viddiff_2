"""
python -m ipdb scripts/20241128_openeval_humanstudy.py
"""

from pathlib import Path
import torch
import ipdb
import click
import json
import os
from datasets import load_dataset
import logging
import torch.nn.functional as F
import sys
import pandas as pd
import numpy as np
import cv2
import yaml
from data import load_viddiff_dataset as lvd

results_dir = Path("scripts/results/") / Path(__file__).stem
results_dir.mkdir(exist_ok=True, parents=True)

dir_configs = Path("lmms/configs/")
dir_evals = Path("lmms/results/")
splits = ["fitness_4fps", "ballsports_5fps",
          "diving_6fps"]  #, "music_1fps", "surgery_1fps"]
models = ["gpt-4o", "gemini-pro", "qwen"]

do_ours = False
if do_ours:
    models = ["viddiff",]
    splits = ['fitness', 'ballsports', 'diving']
    dir_evals = Path('viddiff_method/results')
    dir_configs = Path("viddiff_method/configs/")

dataset = load_dataset("viddiff/VidDiffBench")['test']

n_sample = 1
df_lst = []
for split in splits:
    dataset = lvd.load_viddiff_dataset([split.split("_")[0]])
    df_ = pd.DataFrame(dataset)

    diffs_gt = []
    for d in df_['differences_annotated']:
        diff_gt = {k: v['query_string'] for k, v in d.items() if v is not None}
        diff_gt = dict(sorted(diff_gt.items(), key=lambda x: int(x[0])))
        diffs_gt.append(diff_gt)
    df_["diffs_gt"] = diffs_gt

    for model in models:
        # settings
        if do_ours:
            stem=f"{split}"
            pass 
        else:
            stem = f"{model}_{split}"
            if 'gemini' in model:
                stem += "_1geminifps"

        # with open(dir_configs / f"{stem}.yaml", 'r') as file:
        #     cfg = yaml.safe_load(file)

        # get the model predictions
        dir_preds = dir_evals / stem / "seed_0"
        assert dir_preds.exists()
        with open(dir_preds / "input_predictions.json") as fp:
            preds_dict = json.load(fp)
        assert len(preds_dict) == len(dataset)
        preds_ = [preds_dict[k] for k in dataset['sample_key']]
        preds = []
        for pred_ in preds_:
            pred = {k: v["description"] for k, v in pred_.items()}
            preds.append(pred)
        df = df_.copy()
        df['pred'] = preds

        # get the matching predictions
        with open(dir_preds / "eval_matching.json") as fp:
            preds_match = json.load(fp)
        assert len(preds_match) == len(dataset)
        df['preds_match'] = preds_match

        # save the key things
        df['model'] = model
        df['r0'] = [json.dumps(d, indent=2) for d in df["diffs_gt"]]
        df['r1'] = [json.dumps(d, indent=2) for d in df["pred"]]
        df['r2'] = [json.dumps(d, indent=2) for d in df['preds_match']]

        # subset
        # df_sample = df.groupby(['action']).apply(lambda x: x.sample(1),
        #                                          include_groups=True)
        df_sample = df.sort_values(by='sample_key').groupby('action').head(n_sample).reset_index()

        df_lst.append(df_sample)

df = pd.concat(df_lst)
df = df[["model", "sample_key", "action", "r0", "r1", "r2"]]
df = df.sort_values(["sample_key", "model"])

# skip the actions that are actually repeated
skip_repeats = ['fitness_5', 'fitness_6', 'fitness_7']
df = df[~df['action'].isin(skip_repeats)]

df.to_csv(results_dir / f"matching_{n_sample}psample_{do_ours}.csv")


##### now combine everything
ipdb.set_trace()

df1 = pd.read_csv(results_dir / f"matching_1psample_False.csv")
df2 = pd.read_csv(results_dir / f"matching_1psample_True.csv")
df = pd.concat([df1, df2]).reset_index()

df = df.sort_values(["sample_key", "model"])
# now make the review rows
csv_rows = []
for idx, row in df.iterrows():
    prefix = [row['model'], row['sample_key'], row['action'], row['r1']]
    diffs = json.loads(row['r0'])
    for k, v in diffs.items():
        csv_row = [*prefix, k, v]
        csv_rows.append(csv_row)
        prefix = ["", "", "", ""]  # after the first iteration, get rid of it

df_save = pd.DataFrame(
    csv_rows, columns=['model', 'sample', 'action', 'pred', 'gt_key', 'gts'])
f_save = results_dir / f"matching_to_annotate_all.csv"
print(f_save)
df_save.to_csv(f_save)
ipdb.set_trace()
pass
