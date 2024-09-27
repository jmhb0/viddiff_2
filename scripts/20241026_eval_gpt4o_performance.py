"""
python -m ipdb scripts/20241026_eval_gpt4o_performance.py
"""

import ipdb
import json
from pathlib import Path
import pandas as pd 
from data import load_viddiff_dataset as lvd

all_datasets = []
splits = ['music', 'diving', 'easy', 'fitness', 'surgery', 'demo', 'ballsports',]
dataset = lvd.load_viddiff_dataset(splits)

names = [
    "mode2-gpt-4o_easy_4fps.yaml", "mode2-gpt-4o_ballsports_4fps.yaml",
    "mode2-gpt-4o_fitness_4fps.yaml", "mode2-gpt-4o_diving_4fps.yaml",
    "mode2-gpt-4o_music_2fps.yaml", "mode2-gpt-4o_surgery_2fps.yaml"
]
results_dir_stem = Path("lmms/results/")

all_df = []
for name in names:
	f_df = results_dir_stem / Path(name).stem / "seed_0/df_all_gt_diffs.csv"
	df = pd.read_csv(f_df)
	df = df[df['gt'].isin(['a', 'b'])]
	all_df.append(df)
df = pd.concat(all_df, axis=0)

# correct score
df['1'] = df['gt'] == df['pred']

# lookup the action 
def sample_to_action(dataset):
	lookup = {}
	for row in dataset:
		lookup[row['sample_key']] =  row['action']
	return lookup
lookup_sample_to_action = sample_to_action(dataset)
df['action'] = df["sample_key"].map(lookup_sample_to_action)

# lookup actions
f_save = "scripts/results/20241026_choose_action_split/lookup_action_to_split.json"
with open(f_save, "r") as fp:
	lookup_action_to_split = json.load(fp)
df['split'] = df['action'].map(lookup_action_to_split)

ipdb.set_trace()
pass 

# some metrics
print(df.groupby(['action'])['1'].mean())


