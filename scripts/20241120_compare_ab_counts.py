"""
python -m ipdb scripts/20241120_compare_ab_counts.py
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
from lmms import lmm_utils as lu
import eval_viddiff
from apis import openai_api

def compute_splits(df):
	gts = []
	for idx, row in df.iterrows():
		diff = row['differences_gt']
		letters = [v for k,v in diff.items() if v is not None]
		gts += letters
	letters, cnts = np.unique(gts, return_counts=True)
	print(letters, cnts)
	print(cnts[:2] / cnts[:2].sum())

splits=['ballsports', 'fitness', 'music', 'surgery', 'diving']
dataset = lvd.load_viddiff_dataset(splits=splits)
df = pd.DataFrame(dataset)
print("Overall")
compute_splits(df)

print("easy")
df_ = df[df['split_difficulty']=='easy']
compute_splits(df_)

print("medium")
df_ = df[df['split_difficulty']=='medium']
compute_splits(df_)

print("hard")
df_ = df[df['split_difficulty']=='hard']
compute_splits(df_)
ipdb.set_trace()

print()

# splits = ['fitness']
# dataset = lvd.load_viddiff_dataset(splits=splits)
# compute_splits(dataset)

ipdb.set_trace()
pass
