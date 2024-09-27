"""
python scripts/20241026_choose_difference_splits.py
"""
import ipdb
import click
import json
import os
from datasets import load_dataset
import logging
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(filename)s:%(levelname)s:%(message)s')

sys.path.insert(0, ".")
from data import load_viddiff_dataset as lvd
from lmms import config_utils
from lmms import lmm_utils as lu
import eval_viddiff
from apis import openai_api

results_dir = Path("scripts/results/") / Path(__file__).stem
results_dir.mkdir(exist_ok=True, parents=True)

splits = ("easy", "diving", "fitness", "surgery", "ballsports", "demo",
          "music")
dataset = lvd.load_viddiff_dataset(splits=splits)

all_differences = {}  # across all actions
actions = set()  # just to keep track of which actions are included
for row in dataset:
    action = row['action']
    action_description = row['action_description']
    if action in actions:
        continue

    # get the correct differences and sort correctly
    keys_gt = {k for k, v in row['differences_gt'].items() if v is not None}
    keys_gt = sorted(keys_gt, key=int)
    differences = {
        k: v
        for k, v in row['differences_annotated'].items() if k in keys_gt
    }

    def sort_dict(example_dict):
        return dict(sorted(example_dict.items(),
                           key=lambda item: int(item[0])))

    differences = sort_dict(differences)
    differences_for_prompt = {}

    # extra information
    for k, v in differences.items():
        diff_ = {}
        diff_['name'] = v['name']
        diff_['description'] = v['description']
        # diff_['num_frames'] = v['num_frames']
        diff_['description'] = v['description']
        diff_["split"] = "easy|hard"
        diff_["split_reason"] = "..."

        differences_for_prompt[k] = diff_

    all_differences[action] = {
        "action_description": action_description,
        "differences": differences_for_prompt
    }
# - 'num_frames' which is '1' if the difference could be evaluated by comparing 1 well-chosen frame from each of 'video A' or 'video B'. If multiple frames from each video are needed, it's 'gt_1'.
prompt = """\
I'm designing a benchmark for comparing pairs of videos of the same action.
The task is to examine differences and say whether the statement applies more to "video A" or "video B".


Below we show a dictionary where each element is a single action. 
Each action has a "action_description" describing the action. 
Each action has a dictionary of "differences", where each difference has these keys:
- 'name' for the difference
- 'description' describing the difference
- 'split' which currently says 'easy|hard'
- 'split_reason' which currently says '...'

Your task is to, for each difference
- decide whether the 'split' value is 'easy' or 'hard'. This evaluation judges whether the difference is easier or harder for a person or an AI system to identify whether the difference occurs in that video. 
- justify your choice in 'split_reason'. 

Return the same differences dictionary as json, with the values of 'split' and 'split_reason' populated.


Here are the differences. Again, return this same dictionary with the 'split' and 'split_reason' populated.
{differences}
"""
prompt = prompt.replace("{differences}", f"{json.dumps(all_differences, indent=2)}")
res = openai_api.call_gpt(prompt, json_mode=False, max_tokens=15000)

from lmms.lmm_utils import _remove_trailing_commas_json
# x = json.loads(_remove_trailing_commas_json(res[0]))
x = json.loads(res[0][8:-4]) # manual, whoops
with open(results_dir / "gpt_splits.json", "w") as fp:
    json.dump(x, fp, indent=2)

# turn to pandas 
records = []
for key, value in x.items():
    action_description = value['action_description']
    for diff_key, diff_value in value['differences'].items():
        record = {
            'sample_key': key,
            'split_original' : key.split("_")[0],
            'difference_key': diff_key,
            'action_description': action_description,
            **diff_value
        }
        records.append(record)

# Creating the DataFrame
df = pd.DataFrame(records)
df.to_csv(results_dir / "hardness_splits.csv")
print(df.groupby(["split"])["split"].count())
print(df.groupby(["split_original", "split"])['split'].count())
ipdb.set_trace()
pass

# summarize the 'reasons given'
prompt="""\
I'm designing a benchmark for comparing pairs of videos of the same action.
They will test model's abilities in spotting important differences. 

Below I list many actions and their differences. 
They are classified into splits "easy" or "hard", and they are classified into those two based on "split_reason". 

For each of "easy" and "hard", what are the main reasons given for the classification? 

DIFFERENCES, DIFFICULTY AND REASONS:
{info}
"""
prompt = prompt.replace("{info}", json.dumps(x, indent=2))
res = openai_api.call_gpt(prompt, json_mode=False, max_tokens=15000)
y = res[0]
with open(results_dir / "gpt_spit_reasons.txt", "w") as fp:
    fp.write(y)






