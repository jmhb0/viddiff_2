"""
python -m ipdb scripts/20241122_dataset_statistics.py
"""
import json
import numpy as np
import ipdb
import os
from datasets import load_dataset
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms import config_utils

import eval_viddiff

results_dir = Path(__file__).parent / "results" / Path(__file__).stem
results_dir.mkdir(exist_ok=True)

splits = ['ballsports', 'diving', 'fitness', 'music', 'surgery']
# splits=['diving']
dataset = lvd.load_viddiff_dataset(splits=splits)

## avg video length, total video length
if 1:
    videos = lvd.load_all_videos(dataset, do_tqdm=True, cache=True)
    times_seconds = np.zeros((len(dataset), 2), dtype=float)
    for i in range(len(dataset)):
        vid_pair = [videos[0][i], videos[1][i]]
        for j, vid in enumerate(vid_pair):
            nframes = vid['video'].shape[0]
            fps = vid['fps']
            time = nframes / fps
            times_seconds[i, j] = time
            pass
    print(f"Total minutes   {times_seconds.sum()/60:.3f}")
    print(f"Total hours     {times_seconds.sum()/60**2:.3f}")
    print(f"Average seconds {times_seconds.mean():.3f}")
    print(f"Median  seconds {np.median(times_seconds):.3f}")

    df = pd.DataFrame(dataset)
    lookup_times_split = dict()
    df['time'] = times_seconds.mean(1)
    summarize_actions = df.groupby(
        ['split_difficulty', 'action',
         'action_name'])['time'].agg(mean_time='mean',
                                     total_time_mins=lambda x: x.sum() / 60)
    summarize_actions.to_csv(results_dir / "summarize_actions_time.csv")

    for split_difficulty in ['easy', 'medium', 'hard']:
        print()
        print("*" * 80)
        print(f"split {split_difficulty}")
        idxs = np.where(df['split_difficulty'] == split_difficulty)[0]
        times_seconds_ = times_seconds[idxs]
        print(f"Average seconds {times_seconds_.mean():.3f}")
        print(f"Median  seconds {np.median(times_seconds_):.3f}")
        print(f"Total minutes   {times_seconds_.sum()/60:.3f}")
        print(f"Num pairs       {len(idxs)}")
        lookup_times_split[split_difficulty] = times_seconds_.mean()

## distribution of a/b/c per action
# get the localization annotations
df = pd.DataFrame(dataset)
_df_frames = []
# for each video pair
for idx, row in df.iterrows():
    action = row['action']
    split_difficulty = row['split_difficulty']
    locs = json.loads(row['retrieval_frames'])

    # for each video
    for j in [0, 1]:
        vid_length = videos[j][idx]['video'].shape[0]

        # for each retrieval annotation
        for loc_idx, frame_id in locs[j].items():
            if frame_id is None:
                continue
            # action, location id, frameid, loc_position
            info = [
                split_difficulty, action, loc_idx, frame_id[0],
                frame_id[0] / vid_length
            ]
            _df_frames.append(info)

df_frames = pd.DataFrame(
    _df_frames,
    columns=["split_difficulty", "action", "loc_id", "frameid", "loc_norm"])
# mean_frame_nromalized = df_frames.groupby(['action', 'loc_id'])['loc_norm'].mean().reset_index()
df_frames['std_of_loc_norm'] = df_frames.groupby(['action', 'loc_id'
                                           ])['loc_norm'].transform('std')
locstats_by_split = df_frames.groupby(['split_difficulty'
                                       ])['std_of_loc_norm'].agg(std='mean',
                                                              count="count")
locstats_by_action = df_frames.groupby(['split_difficulty', 'action'
                                        ])['std_of_loc_norm'].agg(std='mean',
                                                               count="count")
locstats_by_action.to_csv(results_dir / "summarize_actions_locs.csv")

# this one is variance of location across a video action - shows that retrieval annotations span the video ... all of the video must be looked at
loc_norm_by_action = df_frames.groupby("action")["loc_norm"].std()
loc_norm_by_split = df_frames.groupby("split_difficulty")["loc_norm"].std()
loc_norm_by_action.to_csv(results_dir / "loc_norm_actions_locs.csv")



### label distribution 
items = []
for idx, row in df.iterrows(): 
	for k, v in row['differences_gt'].items():
		if v is not None:
			item = [row['split_difficulty'], row['action'], k, v]
			items.append(item)
df_diffs = pd.DataFrame(items, columns=['split_difficulty', 'action', "diff_id", "gt"])

# a/b/c
ABC_splits = df_diffs.groupby("split_difficulty")['gt'].value_counts().unstack(fill_value=0).apply(lambda row: '/'.join(map(str, row)), axis=1)
ABC_actions = df_diffs.groupby(["split_difficulty", "action"])['gt'].value_counts().unstack(fill_value=0).apply(lambda row: '/'.join(map(str, row)), axis=1)
count_diffs_splits = df_diffs.groupby("split_difficulty")['gt'].count()
count_diffs_actions = df_diffs.groupby(["split_difficulty", "action"])['gt'].count()
print(count_diffs_splits)
print(count_diffs_splits)
ABC_actions.to_csv(results_dir / "ABC_actions.csv")
count_diffs_actions.to_csv(results_dir / "count_diffs_actions.csv")

ipdb.set_trace()
pass




