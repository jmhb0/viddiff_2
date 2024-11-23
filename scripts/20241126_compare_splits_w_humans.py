"""
python -m ipdb scripts/20241126_compare_splits_w_humans.py
Source sheet https://docs.google.com/spreadsheets/d/1Vx8F_GnEHXgbfRF-ABvBjTQ11smD5hjXouTY4rUqbqI/edit?gid=0#gid=0 
"""
import ipdb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# these are copied from the sheed
# yapf: disable
ranks = [[1,10,1,5],
  [2,3,2,4],
  [3,1,9,1],
  [4,2,8,7],
  [5,11,3,6],
  [6,6,7,9],
  [7,12,6,3],
  [8,14,13,8],
  [9,7,12,11],
  [10,13,15,10],
  [11,18,10,15],
  [12,4,4,2],
  [13,5,5,12],
  [14,8,17,13],
  [15,9,14,14],
  [16,16,18,18],
  [17,15,16,16],
  [18,17,11,17]]
# yapf: enable
df = pd.DataFrame(ranks)
spearman_corr = df.corr(method='spearman')
x = spearman_corr.values
x_nota = x[1:,1:]
print(f"Correlations with LLM in the first col and row, and the others being human generated")
print(spearman_corr)
flattened_no_diag = x[~np.eye(x.shape[0], dtype=bool)].flatten()
flattened_no_diag_noa = x_nota[~np.eye(x_nota.shape[0], dtype=bool)].flatten()

print("mean non-1 correlations across all", flattened_no_diag.mean())
print("mean non-1 correlations across all, excluding 'a'", flattened_no_diag_noa.mean())
print("mean non-1 correlations for 'a' ", x[1:, 0].mean())

ipdb.set_trace()
pass
