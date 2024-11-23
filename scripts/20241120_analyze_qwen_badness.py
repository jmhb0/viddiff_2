"""
python -m ipdb scripts/20241120_analyze_qwen_badness.py
"""
import ipdb 
import json
import numpy as np 
import pandas as pd 
from apis import openai_api
# created by `20241128_openeval_humanstudy.py`
f_save = "scripts/results/20241128_openeval_humanstudy/matching_5psample_False.csv"
f_save = "scripts/results/20241128_openeval_humanstudy/matching_3psample_False.csv"

df = pd.read_csv(f_save)
df = df[df['model'] == 'qwen']

### first, the 'visual appearance' error
if 1:
	diffs_flattened = []
	for idx, row in df.iterrows():
		diff = json.loads(row['r1'])
		diffs_flattened += list(diff.values())

	prompt_template = """
	Here is a string that describes the difference between two videos:
	"{{diff_string}}"

	The string should be about how an action is correctly performed. 
	However it may have a "is_appearance" error. 
	Here, the difference describes the visual appearance, but has nothing to do with HOW the action is performed.
	For example, it may compare the outfits of the people in the videos.
	In this case set "is_appearance" as "true" else "false".

	Return json: {"is_appearance": "true|false"}
	"""
	batch_prompts = [prompt_template.replace("{{diff_string}}", t) for t in diffs_flattened]
	# Proposing differences that are not about how to perform actions, but are visual things like “The person in video a is wearing a blue jacket, while the person in video b is wearing a plaid shirt.”
	# Proposing the exact same difference multiple times, e.g. “The person in video a is performing the exercise with their arms out to the sides, while the person in video b is performing the exercise with their arms out to the sides.”
	# Proposing only a small number of differences.
	# Proposing vague differences that are harder to interpret visually like “The player i
	res = openai_api.call_gpt_batch(batch_prompts, json_mode=True, model="gpt-4o-mini")
	msgs = [r[0] for r in res]
	cost = [r[1] for r in res]
	errs_appearance = [1 if m['is_appearance']=='true' else 0 for m in msgs]
	print("cost", sum(cost))
	if 0:
		for i in range(100):
			print(errs_appearance[i], diffs_flattened[i])
	print('Rate of this error', sum(errs_appearance) / len(errs_appearance))

## next, the 'repeated entry' thing
if 1:
	errs_repeat = []
	counter_questions = 0
	num_unique = 0 
	num_not_unique = 0
	for idx, row in df.iterrows():
		diffs = json.loads(row['r1'])

		diffs_lst = list(diffs.values())
		for d in diffs_lst:
			if diffs_lst.count(d) > 1:
				errs_repeat.append(1)
			else:
				errs_repeat.append(0)
		num_not_unique += len(diffs_lst)
		num_unique += len(set(diffs_lst))
		if len(set(diffs_lst)) < len(diffs_lst):
			counter_questions += 1 

	print("Repeated difference happens for this pcntage of qwen repsonses ", counter_questions/len(df))
	print("Non-repeated diffs", num_unique, "total diffs", num_not_unique)
	print("pcnt of diffs that are actually just copies", (num_not_unique - num_unique) / num_not_unique)

if 0:
	diffs_flattened = []
	for idx, row in df.iterrows():
		diff = json.loads(row['r1'])
		diffs_flattened += list(diff.values())

	prompt_template = """
	Here is a string that describes the difference between two videos:
	"{{diff_string}}"

	The string should be about how an action is correctly performed. 
	However it may have a "is_vague" error. 
	Here, the difference describes something vague that might is difficult to judge impartially. 
	In this case set "is_vague" as "true" else "false".

	An example of "is_vague"="true" example is "The player in video a has a more versatile and adaptable skill set than the player in video b."
	An example of "is_vague"="false" is "The diver in video a has a more powerful takeoff" because it can be assessed from the video pair.

	Return json: {"is_vague": "true|false"}
	"""
	batch_prompts = [prompt_template.replace("{{diff_string}}", t) for t in diffs_flattened]
	# Proposing differences that are not about how to perform actions, but are visual things like “The person in video a is wearing a blue jacket, while the person in video b is wearing a plaid shirt.”
	# Proposing the exact same difference multiple times, e.g. “The person in video a is performing the exercise with their arms out to the sides, while the person in video b is performing the exercise with their arms out to the sides.”
	# Proposing only a small number of differences.
	# Proposing vague differences that are harder to interpret visually like “The player i
	res = openai_api.call_gpt_batch(batch_prompts, json_mode=True, model="gpt-4o-mini")
	msgs = [r[0] for r in res]
	cost = [r[1] for r in res]
	errs_vague = [1 if m['is_vague']=='true' else 0 for m in msgs]
	print("cost", sum(cost))
	if 1:
		for i in range(100):
			print(errs_vague[i], diffs_flattened[i])
	print('Rate of this error', sum(errs_vague) / len(errs_vague))

# combine errors
err = np.array(errs_appearance) + np.array(errs_repeat)
print("Pcnt of predicted errors at the difference level",  (err==0).sum() / len(err))

# the 'is short' error
if 1: 
	counter = 0
	for idx, row in df.iterrows():
		diff_gt = json.loads(row['r0'])
		diff_pred = json.loads(row['r1'])
		n_gt = len(diff_gt)
		n_pred = len(diff_pred)

		# it's allowed up to 1.5 of n_gt, so flag if it doesn't meet half of that
		if n_pred < 0.75*n_gt: 
			counter+=1

	print(f"Questions, with less than half of what they're allowed to predict", counter / len(df))

ipdb.set_trace()
pass










