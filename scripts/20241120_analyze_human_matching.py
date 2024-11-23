"""
python -m ipdb scripts/20241120_analyze_human_matching.py
"""
import ipdb 
import json
import numpy as np 
import pandas as pd 
from pathlib import Path
from apis import openai_api
import requests

results_dir = Path("scripts/results/") / Path(__file__).stem
results_dir.mkdir(exist_ok=True, parents=True)

def download_csv(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print("CSV downloaded and saved at:", output_path)

### prepare the annotated versions
def get_csv(idx):
	lookup_csv = {
		0 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_rIZXgVIWK5EulemnOh3J91lB5hGDZowZxQsSlngdFAWg44tVsCGCdB2wmpcVMQrTCLVhmlb9Wbos/pub?gid=2012975856&single=true&output=csv",
		1 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_rIZXgVIWK5EulemnOh3J91lB5hGDZowZxQsSlngdFAWg44tVsCGCdB2wmpcVMQrTCLVhmlb9Wbos/pub?gid=373168272&single=true&output=csv",
		2 : "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_rIZXgVIWK5EulemnOh3J91lB5hGDZowZxQsSlngdFAWg44tVsCGCdB2wmpcVMQrTCLVhmlb9Wbos/pub?gid=249377123&single=true&output=csv",
	}
	f_annotated = results_dir / f"human_{idx}.csv"
	download_csv(lookup_csv[idx], f_annotated)
	df = pd.read_csv(f_annotated)
	return df 

if 1: 
	df0 = get_csv(0)
	df1 = get_csv(1)
	df2 = get_csv(2)
	x0 = np.array(df0['YOUR PRED']).astype(float).astype(str)
	df1[df1['YOUR PRED']=='na'] = np.nan
	x1 = np.array(df1['YOUR PRED']).astype(float).astype(str)
	df2[df2['YOUR PRED']=='n'] = np.nan
	x2 = np.array(df2['YOUR PRED']).astype(float).astype(str)
	print("Comparing the different labelers")
	print((x0==x1).sum() / len(x0))
	print((x0==x2).sum() / len(x0))
	print((x1==x2).sum() / len(x0))
	# idxs = np.where(x0=='nan')[0]
	# print((x0[idxs]==x1[idxs]).sum() / len(idxs))
	# print((x0[idxs]==x2[idxs]).sum() / len(idxs))
	
df = get_csv(2)

model, sample, pred = None, None, None
for idx, row in df.iterrows():
	if type(row['model']) is not str and np.isnan(row['model']):
		row['model'] = model
	else: 
		model, sample, pred = row[['model', 'sample', 'pred']]
	df.loc[idx, 'model'] = model
	df.loc[idx, 'sample'] = sample
	df.loc[idx, 'pred'] = pred


## now compare against the predictions
f = "scripts/results/20241128_openeval_humanstudy/matching_1psample.csv"
df_preds = pd.read_csv(f)
# lookup_preds = dict(zip(df_preds['sample_key'], [json.loads(x) for x in df_preds['r2']]))


df = df[df['model']!='viddiff']
skip_idxs = []
df['pred_key'] = np.nan
for idx, row in df.iterrows():
	if 'gpt-4o' not in row['model']:
		continue
	# get the prediction from the matching
	try: 
		pred_raw  = json.loads(row['pred'])
		pred_match_ = df_preds[df_preds['sample_key']==row['sample']]
		pred_match_ = pred_match_[pred_match_['model'] == row['model']]
		pred_match = json.loads(pred_match_.iloc[0]['r2'])
		pred_string = pred_match[str(int(row['gt_key']))]['pred_description']

		# get the pred_key
		if pred_string is None: 
			pred_key = np.nan
		else: 
			# go back to the original predictions to recover the pred key
			lookup_pred_to_predkey = {v:k for k,v in json.loads(row['pred']).items()}
			pred_key = float(lookup_pred_to_predkey[pred_string])
		
		df.loc[idx, 'pred_key'] = pred_key

	except: 
		print(f"issue with idx", idx)

x0 = np.array(df['YOUR PRED']).astype(str)
x1 = np.array(df['pred_key']).astype(str)
correct = 0
total = 0
# for i in range(len(x0)):
ipdb.set_trace()
for i in range(len(x0)):
	if not x0[i].isdigit():
		x0[i] = 'nan'
print("acc", (x0==x1).sum() / len(x0))
print("acc", (x0==x1).sum() / len(x0))

df[['YOUR PRED', 'pred_key']]
ipdb.set_trace()
pass









