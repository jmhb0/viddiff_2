import ipdb
import numpy as np
from pathlib import Path
import json
import pandas as pd
import logging
import copy
from apis import openai_api


def eval_viddiff(dataset, predictions_unmatched, eval_mode, seed,
                 n_differences):
    verify_n_differences(predictions_unmatched, n_differences)

    # in open mode, handle the matching
    if eval_mode == 0:
        predictions = do_matching(dataset, predictions_unmatched, seed)
        predictions = test_reverse_statements(predictions, seed, batch_size=20)
    else:
        predictions = predictions_unmatched

    # compute stuff and save to csv
    ipdb.set_trace()
    pass


def verify_n_differences(predictions_unmatched, n_differences):
    for i, preds in enumerate(predictions_unmatched):
        if len(preds) > n_differences:
            raise ValueError(f"Maximum number of allowed differences is {n_differences} "\
                f"but prediction number {i} has {len(preds)}: \n{preds}.")


def do_matching(dataset, predictions_unmatched, seed):
    """ 
    The input 'predictions' have numbered keys, 
    """
    batch_prompts_text = []  # text prompt for doing matching
    # store, for each row, the difference decciptions that actuall have an annotation label (some are None)
    differences_gt_all = []

    for row, pred_unmatched in zip(dataset, predictions_unmatched):
        # first for the gt
        differences_gt = {
            k: v
            for k, v in row['differences_gt'].items() if v is not None
        }
        differences_gt_all.append(differences_gt)
        keys_keep = differences_gt.keys()

        prompt = prompt_open_eval_matching
        prompt = prompt.replace("{action_description}",
                                row['action_description'])
        diff_description_gt = {
            k: v['description']
            for k, v in row['differences_annotated'].items() if k in keys_keep
        }
        diff_description_pred = {
            k: v['description']
            for k, v in pred_unmatched.items()
        }
        prompt = prompt.replace("{differences0}",
                                json.dumps(diff_description_gt))
        prompt = prompt.replace("{differences1}",
                                json.dumps(diff_description_pred))
        prompt = prompt.replace("{dict0_keys}",
                                json.dumps(list(diff_description_gt.keys())))
        prompt = prompt.replace("{dict1_keys}",
                                json.dumps(list(diff_description_pred.keys())))

        batch_prompts_text.append(prompt)

    seeds = [seed for _ in range(len(batch_prompts_text))]
    res = openai_api.call_gpt_batch(batch_prompts_text,
                                    model='gpt-4o-mini',
                                    seeds=seeds)
    cost = sum([b[1] for b in res])
    logging.info(f"Cost for eval difference description matching: ${cost:.4f}")
    matches = [b[0] for b in res]

    ## recover predictions and do the logging
    predictions = []  # matched predictions
    for row, differences_gt, pred_unmatched, match in zip(
            dataset, differences_gt_all, predictions_unmatched, matches):

        # init the stuff to populate
        log_description_match = []
        pred = {}

        # verify that the `matching` output obeys some basic matching properties
        _verify_matching_properties(match, differences_gt, pred_unmatched)

        # iterate over the matches
        pred = {}
        for k, v in match.items():
            pred[k] = {}
            pred[k]['gt_description'] = row['differences_annotated'][k][
                'description']
            pred[k]['pred_description'] = pred_unmatched.get(v,
                                                             {})['description']
            pred[k]['pred'] = pred_unmatched.get(v, {}).get('prediction', None)
            pred[k]['pred_key'] = v

        # save the content
        predictions.append(pred)

    return predictions


def test_reverse_statements(predictions, seed, batch_size=20):
    """ 
    Test if it's opposite. 
    If it is the opposite, then flip the prediction
    We use batch size because performance seems to get worse with too many 
    examples at once. 
    """

    # compile the statements
    statements = []
    for pred in predictions:
        for k, v in sorted(pred.items(), key=lambda x: int(x[0])):
            statements.append(
                [pred[k]['gt_description'], pred[k]['pred_description']])

    # make prompts, based on the batch size
    batch_prompts_text = []
    for i in range(0, len(statements), batch_size):
        prompt = prompt_open_eval_check_opposite
        prompt = prompt.replace("{statements}",
                                str(statements[i:i + batch_size]))
        batch_prompts_text.append(prompt)

    # run the prompts
    res = openai_api.call_gpt_batch(batch_prompts_text)
    cost = sum([r[1] for r in res])
    logging.info(f"Cost for eval on 'is_opposite' statement: ${cost:.4f}")
    is_opposite = []
    for r in res:
        is_opposite += r[0]['results']
    assert all(val in {'0', '1'} for val in is_opposite)

    # put the 'is_opposite' predictions back
    idx = 0
    for pred in predictions:
        for k, v in sorted(pred.items(), key=lambda x: int(x[0])):
            is_op = is_opposite[idx]
            if is_op == '1':
                pred[k]['pred'] = flip_abc(pred[k]['pred'])
            pred[k]['is_opposite'] = is_op
            idx += 1
    assert idx == len(is_opposite)

    return predictions


def _verify_matching_properties(match, differences_gt, pred_unmatched):
    """
    We ask an LLM to perform a matching from "gt differences" to "pred differnces". 
    A gt difference is allowed to be to "None". Check that the matching properties 
    are satisfied/ 
    """

    # each 'gt difference' is in the returned object
    if set(match.keys()) != set(differences_gt.keys()):
        raise ValueError(
            f"LLM error. Attempted matching [{match}] doesn't have the right difference " \
            "keys [{differences_gt}]. "\
            "Change the seed passed in to eval_viddiff() function and retry matching")

    # each 'predicted difference' that is matched appears no more than once
    match_vals = [v for k, v in match.items() if v.isdigit()]
    if len(match_vals) != len(match_vals):
        raise ValueError(
            f"LLM error. There are duplicates in values of matches dict: [{matches}]. "
            "Each predicted difference can be matched to at most one gt difference" \
            "Change the seed passed in to eval_viddiff() function and retry matching")

    # each 'predicted difference' from the `matches` is actually a predicted differences
    if not all([m in pred_unmatched.keys() for m in match_vals]):
        raise ValueError(
            f"LLM error. The matches dict: [{match}] has values that are not one of "\
            f" the keys in the predictions: {pred_unmatched.keys()}"
            "Change the seed passed in to eval_viddiff() function and retry matching")

    return


def flip_abc(r):
    return {'a': 'b', 'b': 'a', 'c': 'c'}[r]


# prompt_open_eval_matching = """\
# I am analyzing videos of people performing an action with description: "{action_description}".

# A 'difference' descibes a way that two people might perform the action differently.
# Each difference has a name, a description string that describes what might be different between a pair of videos.

# Here is differences dict 0:
# {differences0}

# Here is differences dict 1:
# {differences1}

# Perform a matching from dict 0 to 1.
# That is, for each item in dict 0, find a good match in dict 1, but each item in dict 1 can only be used once.
# Only match the items if the description string matches closely. If there is no good match, return "None".
# Focus on what visual features are been described. If the words describe a similar thing, but the word choice is different, it may still be a good match.

# Return a json that uses the dict keys to do the matching.
# The keys of the response are all of the keys from dict 0: {dict0_keys}
# The values are a single key from dict 1 or "None". The dict2 keys are: {dict1_keys}
# For example:
# {
#     "0" : "3",
#     "1" : "None",
#     "2" : "1",
#     "3" : "5",
#     ...
# }
# """
prompt_open_eval_matching = """\
You are analyzing videos of people performing a specific action described as "{action_description}."

In this task, a "difference" refers to how two people might perform the same action in distinct ways. 
Each difference consists of:
- A name (key) and
- A description that explains how the two performances differ visually.

You are provided with two dictionaries:
- Dictionary 0: {differences0}
- Dictionary 1: {differences1}

Your task is to match the differences from Dictionary 0 to Dictionary 1. Here's how:
1. For each entry in Dictionary 0, find the best match in Dictionary 1.
2. Each item in Dictionary 1 can only be used once.
3. Only match entries if their description strings are visually similar, even if the word choices differ. If no suitable match exists, return "None."

Your output should be a JSON object where:
- The keys are from Dictionary 0: {dict0_keys}.
- The values are a key from Dictionary 1 or "None" if no match is found. 
- The available keys from Dictionary 1 are: {dict1_keys}.

Example output format:
{
    "0": "3",
    "1": "None",
    "2": "1",
    "3": "5",
    ...
}
"""

prompt_open_eval_check_opposite = """\
You will be given pairs of statements to compare. 
Your task is to determine whether each pair of statements is equivalent or opposite in meaning.

Instructions:

1. For each pair of statements, analyze their logical relationship.
2. Categorize each pair as follows:
   - Return '0' if the statements are equivalent or very similar in meaning. E.g. "X is bigger than Y" and "X is larger than Y" are similar.
   - Return '1' if the statements are directly opposite in meaning. E.g. "X is bigger than Y" is opposite to "X is smaller than Y".

Important Notes:
- For '1' responses, the statements should be true opposites, not just differences in degree.
  Example: "X is much bigger than Y" and "X is slightly bigger than Y" should be categorized as '0', not '1'.

Output Format:
Provide your response as a JSON object containing an array of '0' or '1' values:
{"results" : ["1", "0", "0", ...]}

Input:
The list of statement pairs to analyze will be provided in the following format:

{statements}
"""
