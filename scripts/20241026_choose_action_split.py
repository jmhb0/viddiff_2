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

splits = ("diving", "fitness", "surgery", "ballsports", "music")
dataset = lvd.load_viddiff_dataset(splits=splits)
videos0, videos1 = lvd.load_all_videos(dataset, do_tqdm=True)

videos = videos0 + videos1
actions = [row['action'] for row in dataset] * 2
nframes = [len(v['video']) for v in videos]
fps = [v['fps'] for v in videos]
seconds = [n / f for (n, f) in zip(nframes, fps)]

df_stats = pd.DataFrame({"action": actions, "seconds": seconds})
action_to_seconds = df_stats.groupby(['action'])['seconds'].median().to_dict()
# I need the video length
actions = set(dataset['action'])
action_cnts = {}

# make the object
all_actions = {}  # across all actions
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

        differences_for_prompt[k] = diff_

    all_actions[action] = {
        "action_description": action_description,
        "differences": differences_for_prompt,
        "average_seconds_per_video": f"{action_to_seconds[action]:.1f}",
        "split": "easy|medium|hard",
        "split_reason": "...",
    }

#
prompt = """\
I'm designing a benchmark for comparing pairs of videos of the same action. 
We have many actions and each action has a list of differences we look for.
The benchmark's task is to examine differences and say whether the statement applies more to "video A" or "video B".

Below we show a dictionary where each element is a single action. 
Each action has an "action_description" describing the action. 
It also has "average_seconds_per_video", for the median length of videos in seconds.
Each action has a dictionary of "differences", where each difference has these keys:
- 'name' for the difference
- 'description' describing the difference

Finally, for each action, there are two unfinished options:
- 'split' which currently says 'easy|medium|hard'
- 'split_reason' which currently says '...'
Your task is to fill in these values: 
- Decide whether the 'split' value is 'easy', 'medium' or 'hard'. This evaluation judges the difficulty of performing actionn difference comparison for all differences within an action. Having a high number of actions should not be considered as criteria for difficulty.
- Justify your choice in 'split_reason'. 

Return the same dictionary as json, with the values of 'split' and 'split_reason' populated.
Here are the actions.
{actions}
"""
prompt = prompt.replace("{actions}", f"{json.dumps(all_actions, indent=2)}")
res = openai_api.call_gpt(
    prompt,
    json_mode=True,
    max_tokens=15000,
)
x = res[0]

ipdb.set_trace()

# from lmms.lmm_utils import _remove_trailing_commas_json
# x = json.loads(_remove_trailing_commas_json(res[0]))
# x = json.loads(res[0][8:-4]) # manual, whoops

with open(results_dir / "gpt_splits.json", "w") as fp:
    json.dump(x, fp, indent=2)

x_ = {
    k: {
        'action_description': v['action_description'],
        'split': v['split'],
        'split_reason': v['split_reason'],
    }
    for k, v in x.items()
}
with open(results_dir / "gpt_splits_abbreviated.json", "w") as fp:
    json.dump(x_, fp, indent=2)

# create a simple dict lookup
lookup = { k : v['split'] for k, v in x.items()}
with open(results_dir / "lookup_action_to_split.json", "w") as fp:
    json.dump(lookup, fp, indent=2)

# 
lookup_splits = dict(easy=[], medium=[], hard=[])
for k, v in x.items():
    lookup_splits[v['split']].append(v['action_description'])
with open(results_dir / "lookup_split_to_actiondescription.json", "w") as fp:
    json.dump(lookup_splits, fp, indent=2)



# summarize the 'reasons given'
prompt = """\
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
