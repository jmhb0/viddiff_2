# VidDiff 


## Benchmark eval 
Download the dataset (link). 


## Running eval
In `eval_diff.py`, run 
```
dataset = lvd.
metrics = eval_viddiff.eval_viddiff(
	dataset,
    predictions_unmatched=predictions,
    eval_mode=0,
    seed=0,
    n_differences=10,
    results_dir="results")
```

The structure for `predictions_unmatched`:
```
[
  {
  	// list element i is a dict of difference predictions for sample i
    "numericKey1": {
      // prediction details for one difference
      "description": "..." // A description of the predicted difference",
      "pred": "a|b" // Whather the description is more true of video a or b
    },
    "numericKey2": {
      // Another difference prediction for the same smaple
      // ... same structure as above ...
    }
    // There can be multiple differences per sample
  },
  {
    // Another set of observations
    // ... same structure as above ...
  }
  ... 
]
```


## LLM evaluation for matching, and possible errors
TODO: explain how these can be handled 

Mention the openai 'overwrite_cache'