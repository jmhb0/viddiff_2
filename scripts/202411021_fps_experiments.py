"""
python -m ipdb scripts/202411021_fps_experiments.py
"""

import ipdb
import json
from pathlib import Path
import pandas as pd
from data import load_viddiff_dataset as lvd

results_dir = Path(__file__).parent / "results" / Path(__file__).stem
results_dir.mkdir(exist_ok=True)

mode = 2
compare = 'claude'
print(f"Running mode ", mode)
print("comparing model ", compare)
print()

all_datasets = []
splits = ['fitness']
dataset = lvd.load_viddiff_dataset(splits)

### gpt-4o results
if compare == 'gpt':
	if mode == 2:
		names = [
			"mode2-gpt-4o_fitness_1fps.yaml", "mode2-gpt-4o_fitness_2fps.yaml", 
			"mode2-gpt-4o_fitness_4fps.yaml", "mode2-gpt-4o_fitness_8fps.yaml"
		]
	else:
		raise 
elif compare == 'gemini':
	if mode == 2:
		names = [
		"mode2-gemini-pro_fitness_1fps_1geminifps.yaml", 
		"mode2-gemini-pro_fitness_2fps_1geminifps.yaml",
		"mode2-gemini-pro_fitness_4fps_1geminifps.yaml",
		"mode2-gemini-pro_fitness_8fps_1geminifps.yaml"			
		]
elif compare == 'claude':
	if mode == 2:
		names = [
		"lmms/configs/mode2-claudesonnet_fitness_1fps.yaml",
		"lmms/configs/mode2-claudesonnet_fitness_2fps.yaml",
		"lmms/configs/mode2-claudesonnet_fitness_4fps.yaml",
		"lmms/configs/mode2-claudesonnet_fitness_8fps.yaml",
		]
	else:
		raise 
else: 
	raise 


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

	# get the action description
	lookup_action_to_description = {}
	for row in dataset:
		lookup_action_to_description[row['action']] = row['action_description']
	df['action_description'] = df['action'].map(lookup_action_to_description)

	return df

results_dir_stem_lmm = Path("lmms/results/")
for name in names:
	print()
	print(name)
	f_df = results_dir_stem_lmm / Path(
		name).stem / "seed_0/df_all_gt_diffs.csv"
	df = pd.read_csv(f_df)
	df = df[df['gt'].isin(['a', 'b'])]
	f_preds = results_dir_stem_lmm / Path(
		name).stem / "seed_0/input_predictions.json"
	with open(f_preds, 'r') as fp:
		preds = json.load(fp)

	df_gpt = df
	# add columns
	df_gpt = add_cols_to_df(df_gpt)
	df_gpt['1'] = df_gpt['gt'] == df_gpt['pred']
	df_easy = df_gpt[df_gpt['split'] == 'easy']

	print("Acc easy split ", df_easy['1'].mean())

ipdb.set_trace()
pass

# some metrics
print("ours")
print(df.groupby(['split_us'])['1_us'].mean())
print("Comparing agains ", compare)
print(df.groupby(['split_gpt'])['1_gpt'].mean())

score_gpt = df.groupby(['sample_key', 'split_us'])['1_gpt'].mean()

# any sample keys missinf from each other
diffs_ranked_us = df_us.groupby(
	['split', 'action', 'action_description', 'gt_key',
	 'gt_description'])['1'].agg(['mean', 'count']).sort_values(by='mean')
diffs_ranked_gpt = df_gpt.groupby(
	['split', 'action', 'action_description', 'gt_key',
	 'gt_description'])['1'].agg(['mean', 'count']).sort_values(by='mean')
diffs_ranked_us.to_csv(results_dir / "diffs_ranked_ours.csv")
if compare == 'gpt':
	f_save = results_dir / "diffs_ranked_gpt.csv"
else:
	f_save = results_dir / "diffs_ranked_gemini.csv"
diffs_ranked_gpt.to_csv(f_save)
(diffs_ranked_us - diffs_ranked_gpt).sort_values(by='mean').to_csv(
	results_dir / "diffs_ranked_ours_minus_gpt.csv")

if mode == 0:

	df_us['error_recall'] = df_us['pred'].isna()
	df_gpt['error_recall'] = df_gpt['pred'].isna()
	df_us['error_flip'] = ~(df_us['pred'].isna()) & (df_us['pred'] !=
													 df_us['gt'])
	df_gpt['error_flip'] = ~(df_gpt['pred'].isna()) & (df_gpt['pred'] !=
													   df_gpt['gt'])

	print('recall errors us')
	print(df_us.groupby('split')['error_recall'].mean())

	print('flip errors us')
	print(df_us.groupby('split')['error_flip'].mean())

	print('recall errors gpt')
	print(df_gpt.groupby('split')['error_recall'].mean())

	print('flip errors gpt')
	print(df_gpt.groupby('split')['error_flip'].mean())

# also count the number of predictions
cnts = [len(p) for p in all_preds]
count = sum(cnts)
n_diff_p_pred = count / len(cnts)

### now compute the categories
df_us['category'] = [a[0] for a in df_us['sample_key'].str.split("_")]
print("Num samples per category")
print(df_us.groupby(['category'])['sample_key'].nunique())

df = pd.DataFrame(dataset)
ipdb.set_trace()
pass
