# VidDiff eval
This page explains running eval for the Viddiff benchmark, hosted on Huggingface [here](). It was proposed in "Video Action Differencing". 

The 

The paper also proposed Viddiff method, which is in `viddiff_method`. To run it, look at [this README](viddiff_method/README.md). 

## Get the dataset
TODO: 
- Link to HF. 
- Reproduce the key things here, but link out to how to download the videos. 
- That has instructions on how to load these extra files: they are already in this repo.
- About the video caching, and how it's on by default. 
- Different dataset splits. 
- do explain the form of the 'videos' tuple. 

## Running eval
In `eval_diff.py`, after loading the dataset and running predictions, run:
```
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
	// list element i is a dict of difference predictions for sample i
	{
		"numericKey1": {
			// prediction details for one difference
			"description": "..." // A description of the predicted difference",
			"pred": "a|b" // Whather the description is more true of video a or b
		},
		"numericKey2": {
			// Another difference prediction for the same smaple
			// ... same structure as above ...
		}
		// There can be multiple difference predictions per sample
	},
	{
		// Another set of observations
		// ... same structure as above ...
	}
	... 
]
```

For example, here are predictions for a 3-element dataset. 
```
[
	// an example
]
```

## LLM eval 
The eval file makes some api cals to openaiAPI. Need to set the OpenAI Api key 


## LLM evaluation for matching, and possible errors
TODO: explain how these can be handled 

Mention the openai 'overwrite_cache'


## Video-LMM baselines 
Some baselines are implemented in `lmms`. 
- Which models. 
- Different video representations.
- Same prompt except for description of how the videos are represented. 
- They all do automatic caching. 


## VidDiff method 
The Viddiff method is in `viddiff_method`. To run it, look at [this README](viddiff_method/README.md). 

## Citation 
TODO: copy what's in the HF repo. 


## TODO somewhere 
- Discuss the `fps`. This is not a property of the data, but is a property of the standard implementation. There are functions for subsampling in the `data/` util files, but different methods may want to subset the videos differently.
- pip install and so on 

