"""
python -m ipdb scripts/20241122_compare_lmms_action_level.py
"""

import ipdb
import json
from pathlib import Path
import pandas as pd
from data import load_viddiff_dataset as lvd

results_dir = Path(__file__).parent / "results" / Path(__file__).stem
results_dir.mkdir(exist_ok=True)

f_save = "scripts/results/20241026_choose_action_split/lookup_action_to_split.json"
with open(f_save, "r") as fp:
    lookup_action_to_split = json.load(fp)

splits = ['ballsports', 'diving', 'fitness', 'music', 'surgery']
dataset = lvd.load_viddiff_dataset(splits)

def get_dfs(mode, model):
	if model == 'gpt':
	    if mode == 0:
	        names = [
	            "gpt-4o_ballsports_5fps.yaml", "gpt-4o_fitness_4fps.yaml",
	            "gpt-4o_diving_6fps.yaml", "gpt-4o_music_2fps.yaml",
	            "gpt-4o_surgery_2fps.yaml"
	        ]
	    elif mode == 2:
	        names = [
	            "mode2-gpt-4o_ballsports_4fps.yaml",
	            "mode2-gpt-4o_fitness_4fps.yaml", "mode2-gpt-4o_diving_4fps.yaml",
	            "mode2-gpt-4o_music_2fps.yaml", "mode2-gpt-4o_surgery_2fps.yaml"
	        ]
	elif model == "gemini":
	    if mode == 0:
	        names = [
	            "gemini-pro_fitness_4fps_1geminifps.yaml",
	            "gemini-pro_ballsports_5fps_1geminifps.yaml",
	            "gemini-pro_diving_6fps_1geminifps.yaml",
	            "gemini-pro_surgery_2fps_1geminifps.yaml"
	        ]
	    elif mode == 2:
	        names = [
	            "mode2-gemini-pro_fitness_4fps_1geminifps.yaml",
	            "mode2-gemini-pro_ballsports_5fps_1geminifps.yaml",
	            "mode2-gemini-pro_diving_6fps_1geminifps.yaml",
	            "mode2-gemini-pro_music_2fps_1geminifps.yaml",
	            "mode2-gemini-pro_surgery_2fps_1geminifps.yaml"
	        ]
	elif model == "qwen":
	    if mode == 0:
	        names = [
	            "qwen_ballsports_5fps.yaml",
	            "qwen_diving_6fps.yaml",
	            "qwen_fitness_4fps.yaml",
	            "qwen_music_2fps.yaml",
	            "qwen_surgery_2fps.yaml ",
	        ]
	    elif mode == 2:
	        names = [
	            "mode2_qwen_ballsports_5fps.yaml",
	            "mode2_qwen_diving_6fps.yaml",
	            "mode2_qwen_fitness_4fps.yaml",
	            "mode2_qwen_music_2fps.yaml",
	            "mode2_qwen_surgery_2fps.yaml",
	        ]
	elif model == "claude":
	    if mode == 0:
	        names = [
	            "claudesonnet_ballsports_5fps.yaml",
	            "claudesonnet_diving_6fps.yaml",
	            "claudesonnet_fitness_4fps.yaml",
	            "claudesonnet_music_2fps.yaml",
	            "claudesonnet_surgery_2fps.yaml",
	        ]
	    elif mode == 2:
	        names = [
	            "mode2-claudesonnet_ballsports_4fps.yaml",
	            "mode2-claudesonnet_diving_4fps.yaml",
	            "mode2-claudesonnet_fitness_4fps.yaml",
	            "mode2-claudesonnet_music_2fps.yaml",
	            "mode2-claudesonnet_surgery_2fps.yaml",
	        ]
	elif model == "llavavideo":
	    if mode == 0:
	        names = [
	        "llavavideo_ballsports_5fps.yaml",
	        "llavavideo_diving_6fps.yaml",
	        "llavavideo_fitness_4fps.yaml",
	        "llavavideo_music_2fps.yaml",
	        "llavavideo_surgery_2fps.ya",
	        ]
	    elif mode == 2:
	        names = [
	        "mode2-llavavideo_surgery_2fps.yaml",
	        "mode2-llavavideo_fitness_4fps.yaml",
	        "mode2-llavavideo_ballsports_4fps.yaml",
	        "mode2-llavavideo_music_2fps.yaml",
	        "mode2-llavavideo_diving_6fps.yaml", 
	        ]
	else:
	    raise ValueError()

	results_dir_stem_lmm = Path("lmms/results/")
	all_df = []
	all_preds = []
	for name in names:
	    f_df = results_dir_stem_lmm / Path(
	        name).stem / "seed_0/df_all_gt_diffs.csv"
	    df = pd.read_csv(f_df)
	    df = df[df['gt'].isin(['a', 'b'])]
	    all_df.append(df)
	    f_preds = results_dir_stem_lmm / Path(
	        name).stem / "seed_0/input_predictions.json"
	    with open(f_preds, 'r') as fp:
	        preds = json.load(fp)
	    all_preds += preds

	df = pd.concat(all_df, axis=0)
	df = add_cols_to_df(df) # compute extra thingos

	df['split'] = df['action'].map(lookup_action_to_split)
	if mode != 0:
	    df = df.fillna('b')
	return df, all_preds

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

mode=2
model='gpt'
df_gpt, _ =  get_dfs(mode=mode, model='gpt')
df_gemini, _ =  get_dfs(mode=mode, model='gemini')
df_claude, _ =  get_dfs(mode=mode, model='claude')

ipdb.set_trace()
pass

df = df_gpt.copy()



