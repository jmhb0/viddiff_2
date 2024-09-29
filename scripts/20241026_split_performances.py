"""
python -m ipdb scripts/20241026_split_performances.py
"""

import ipdb
import json
from pathlib import Path
import pandas as pd
from data import load_viddiff_dataset as lvd

mode = 0
print(f"Running mode ", mode)
# 'compare' = 'gpt|gemini'
# compare = 'gemini'
compare = 'gpt'
print()
print("comparing moddel ", compare)

all_datasets = []
splits = ['music', 'diving', 'fitness', 'surgery', 'demo', 'ballsports']
dataset = lvd.load_viddiff_dataset(splits)

### gpt-4o results
if compare == 'gpt':
    if mode == 2:
        names = [
            "mode2-gpt-4o_ballsports_4fps.yaml",
            "mode2-gpt-4o_fitness_4fps.yaml", "mode2-gpt-4o_diving_4fps.yaml",
            "mode2-gpt-4o_music_2fps.yaml", "mode2-gpt-4o_surgery_2fps.yaml"
        ]
    elif mode == 0:
        names = [
            "gpt-4o_ballsports_5fps.yaml", "gpt-4o_fitness_4fps.yaml",
            "gpt-4o_diving_6fps.yaml", "gpt-4o_music_2fps.yaml",
            "gpt-4o_surgery_2fps.yaml"
        ]
elif compare == "gemini":
    if mode == 0:
        raise NotImplementedError()
    elif mode == 2:
        names = [
            "mode2-gemini-pro_fitness_4fps_1geminifps.yaml",
            # "mode2-gemini-pro_ballsports_5fps_1geminifps.yaml",
            "mode2-gemini-pro_diving_6fps_1geminifps.yaml",
            "mode2-gemini-pro_music_2fps_1geminifps.yaml",
            # "mode2-gemini-pro_surgery_2fps_1geminifps.yaml"
        ]
else:
    raise ValueError()

results_dir_stem_lmm = Path("lmms/results/")
all_df = []
for name in names:
    f_df = results_dir_stem_lmm / Path(
        name).stem / "seed_0/df_all_gt_diffs.csv"
    df = pd.read_csv(f_df)
    df = df[df['gt'].isin(['a', 'b'])]
    all_df.append(df)
df_gpt = pd.concat(all_df, axis=0)

### our results
if mode == 2:
    names = [
        "eval2_ballsports.yaml", "eval2_diving.yaml", "eval2_fitness.yaml",
        "eval2_music.yaml", "eval2_surgery.yaml"
    ]
elif mode == 0:
    names = [
        "ballsports.yaml",
        "fitness.yaml",
        "diving.yaml",
        "music.yaml",
        "surgery.yaml",
    ]

###

results_dir_stem_viddiff = Path("viddiff_method/results")
all_df = []
for name in names:
    f_df = results_dir_stem_viddiff / Path(
        name).stem / "seed_0/df_all_gt_diffs.csv"
    df = pd.read_csv(f_df)
    df = df[df['gt'].isin(['a', 'b'])]
    all_df.append(df)
df_us = pd.concat(all_df, axis=0)

#

# filter
df_us = df_us[df_us.set_index(['sample_key', 'gt_key']).index.isin(
    df_gpt.set_index(['sample_key', 'gt_key']).index)].copy()
df_gpt = df_gpt[df_gpt.set_index(['sample_key', 'gt_key']).index.isin(
    df_us.set_index(['sample_key', 'gt_key']).index)].copy()

# df_us = df_us[~df_us['sample_key'].isin(skip)].copy()
# df_gpt = df_gpt[~df_gpt['sample_key'].isin(skip)].copy()

# drop duplicates
# df_us = df_us.drop_duplicates(subset=['sample_key', 'gt_key'], keep='last').copy()
# df_gpt = df_gpt.drop_duplicates(subset=['sample_key', 'gt_key'], keep='last').copy()


# func for columns
def add_cols_to_df(df):
    # correct score
    df['1'] = df['gt'] == df['pred']

    # lookup the action
    def sample_to_action(dataset):
        lookup = {}
        for row in dataset:
            lookup[row['sample_key']] = row['action']
        return lookup

    lookup_sample_to_action = sample_to_action(dataset)
    df['action'] = df["sample_key"].map(lookup_sample_to_action)

    # lookup split
    f_save = "scripts/results/20241026_choose_action_split/lookup_action_to_split.json"
    with open(f_save, "r") as fp:
        lookup_action_to_split = json.load(fp)
    df['split'] = df['action'].map(lookup_action_to_split)

    return df


# add columns
df_us = add_cols_to_df(df_us)
df_gpt = add_cols_to_df(df_gpt)

# some metrics
print("ours")
print(df_us.groupby(['split'])['1'].mean())
print("gpt")
print(df_gpt.groupby(['split'])['1'].mean())

# some other stuff
df = pd.merge(df_us,
              df_gpt,
              on=['sample_key', 'gt_key'],
              how='inner',
              suffixes=('_us', '_gpt'))

score_us = df.groupby(['sample_key'])['1_us'].mean()
score_gpt = df.groupby(['sample_key'])['1_gpt'].mean()
diff = score_gpt - score_us

# any sample keys missinf from each other

set(df_gpt['sample_key'])
ipdb.set_trace()
pass
