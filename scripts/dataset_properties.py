"""
python -m ipdb scripts/dataset_properties.py
"""
import ipdb
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

import sys 
sys.path.insert(0, "")
from data import load_viddiff_dataset as lvd

N_DIFF_MULTIPLIER = 1.5
N_DIFF_MIN = 5

results_dir = Path("scripts/results/dataset_properties")
results_dir.mkdir(exist_ok=True, parents=True)

if 1:
    metas = {}
    for split in ['music', 'diving', 'easy', 'fitness', 'surgery', 'demo', 'ballsports',]:
        print(split)
        meta = {}
        dataset = lvd.load_viddiff_dataset([split])
        ipdb.set_trace()
        videos0, videos1 = lvd.load_all_videos(dataset, do_tqdm=True)
        videos = videos0 + videos1

        # make a histogram of video lengths, and put fps in the title
        nframes = [len(v['video']) for v in videos]
        fps = [v['fps'] for v in videos]
        seconds = [n / f for (n, f) in zip(nframes, fps)]
        heights = [v['video'].shape[1] for v in videos]
        widths = [v['video'].shape[2] for v in videos]

        # count actions 
        num_actions = len(set(dataset['action']))
        unique_variations = set()
        for row in dataset:
            action = row['action']
            variation_names = { f"{action}_{v['name']}" for k,v in row['differences_annotated'].items() if v is not None}
            unique_variations = unique_variations | variation_names
        num_variations = len(unique_variations)




        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Split: {split}, num pairs: {len(videos)//2}', fontsize=16)

        ax1.hist(seconds, bins=20, edgecolor='black')
        ax1.set_xlabel('Seconds')
        ax1.set_ylabel('Frequency')
        ax2.hist(nframes, bins=20, edgecolor='black')
        ax2.set_xlabel('Number of Frames')
        ax2.set_ylabel('Frequency')
        ax3.hist(fps, bins=20, edgecolor='black')
        ax3.set_xlabel('Frames per Second')
        ax3.set_ylabel('Frequency')
        ax4.hist(heights, bins=20, edgecolor='black')
        ax4.set_xlabel('Frame height')
        ax4.set_ylabel('Frequency')
        ax5.hist(widths, bins=20, edgecolor='black')
        ax5.set_xlabel('Frame height')
        ax5.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(results_dir / f"video_info_{split}.png")
        plt.close()

        actions = dataset['action']
        n_differences_all = {}
        n_differences_all_target = {}
        for action in sorted(set(actions)):
            row = dataset.filter(lambda x: x['action']==action)[0]
            differences_gt = row['differences_gt']
            n_differences = len({k for k,v in differences_gt.items() if v is not None})
            n_differences_all[action] = n_differences
            n_differences_all_target[action] = max(N_DIFF_MIN, int(n_differences*N_DIFF_MULTIPLIER))

        meta['seconds'] = np.median(seconds)
        meta['nframes'] = np.median(nframes)
        meta['fps'] = np.median(fps)
        meta['height'] = np.median(heights)
        meta['width'] = np.median(widths)
        meta['n_differences_all'] = n_differences_all
        meta['n_differences_all_target'] = n_differences_all_target
        meta['num_samples'] = len(dataset)
        meta['num_actions'] = num_actions
        meta['num_samples_p_action'] = len(dataset) / num_actions
        meta['num_dataset'] = len(dataset)
        meta['num_variations'] = num_variations

        metas[split] = meta

    # save everything
    with open(results_dir / "meta.json", "w") as fp:
        json.dump(metas, fp, indent=4)

    # target_differences 
    n_diffs = {k : v['n_differences_all_target'] for k, v in metas.items()}
    f_save = results_dir / "n_differences.json"
    print(str(f_save))
    with open(f_save, "w") as fp:
        json.dump(n_diffs, fp, indent=4)

# now let's understand the 'easy' dataset 
dataset = lvd.load_viddiff_dataset(['easy'])
videos0, videos1 = lvd.load_all_videos(dataset, do_tqdm=True)
diffs = dataset['differences_gt']
for i in range(len(diffs)):
    diffs[i] = {k:v for k,v in diffs[i].items() if v is not None}

ipdb.set_trace()        
pass 


