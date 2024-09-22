import ipdb
import numpy as np
from pathlib import Path
import json
import pandas as pd
import logging
import copy
from datasets import Dataset
from pathlib import Path
import math

from apis import openai_api


def eval_viddiff(dataset: Dataset,
                 predictions_unmatched: list[dict[str, dict]],
                 eval_mode: int,
                 seed: int,
                 n_differences: list[int],
                 diffs_already_matched: bool = False,
                 results_dir=None):

    validate_prediction_schema(predictions_unmatched, n_differences)

    # first handle the matching
    if not diffs_already_matched:
        predictions = do_matching(dataset, predictions_unmatched, seed)
        predictions = test_reverse_statements(predictions, seed, batch_size=6)
    else:
        predictions = predictions_unmatched
        predictions = add_extra_details(dataset, predictions)

    # combine the predictions into a dataframe and compute some metrics
    df = make_eval_df(dataset, predictions)
    metrics = compute_metrics(df)

    # logging
    log(df, metrics, results_dir)
    print(metrics)
    ipdb.set_trace()

    return metrics


def compute_metrics(df_notfiltered, results_dir=None):
    """
    Compute the metrics, where we only consider samples with 'a' or 'b 
    """
    # in standard mode, only want rows where the gt difference is 'a' or 'b'
    df = df_notfiltered[df_notfiltered['gt'].isin(['a', 'b'])].copy()
    recall = (df['pred'] == df['gt']).sum() / len(df)

    # error types - no match, or wrong prediction
    df['err_nomatch'] = df['pred'].isna()
    err_nomatch = df['err_nomatch'].mean()
    df['err_flippedpred'] = (df['pred'] != df['gt']) & (df['pred'].isin(
        ['a', 'b']))
    err_flippedpred = df['err_flippedpred'].mean()
    df['err_is_c'] = (df['pred'] == 'c')
    err_is_c = df['err_is_c'].mean()
    # err = ((x['gt'] == 'a') & (x['pred'] == 'b')) | ((x['gt'] == 'b') & (x['pred'] == 'a'))

    assert math.isclose(err_flippedpred + err_nomatch + err_is_c, 1 - recall)

    metrics = dict(recall=recall,
                   err_nomatch=err_nomatch,
                   err_flippedpred=err_flippedpred,
                   err_is_c=err_is_c)

    return metrics


def log(df, metrics, results_dir):
    if results_dir is None:
        return

    results_dir = Path(results_dir)

    # log metrics
    with open(results_dir / "eval_metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=4)

    # log matching only
    log_items = []
    for sample_key in df['sample_key'].unique():
        df_ = df[df['sample_key'] == sample_key]
        log_item = df_.set_index('gt_key')[[
            'sample_key', 'gt_description', 'pred_description', "is_opposite"
        ]].to_dict('index')
        log_items.append(log_item)
    with open(results_dir / "eval_matching.json", 'w') as fp:
        json.dump(log_items, fp, indent=4)

    # log the full csv
    df.to_csv(results_dir / "df_all_gt_diffs.csv")

    # log the csv for only when gt!='c'
    df_ = df[df['gt'] == 'c'].copy()
    df_.to_csv(results_dir / "df_gt_positive_diffs.csv")


def validate_prediction_schema(predictions_unmatched: list[dict],
                               n_differences: list[int]):
    # check the max number of differences was not exceeded
    for i, preds in enumerate(predictions_unmatched):
        if len(preds) > n_differences[i]:
            raise ValueError(f"Maximum number of allowed differences is {n_differences[i]} "\
                f"but prediction number {i} has {len(preds)}: \n{preds}.")

    # check that each prediction is one of 'a' or 'b'
    for pred in predictions_unmatched:
        for k, v in pred.items():
            assert k.isdigit()
            assert 'prediction' in v.keys()
            assert 'description' in v.keys()
            assert v['prediction'] in ('a', 'b', 'c')


def do_matching(dataset, predictions_unmatched, seed):
    """ 
    The input 'predictions' have numbered keys. 
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
                                    overwrite_cache=True,
                                    seeds=seeds)
    cost = sum([b[1] for b in res])
    logging.info(f"Cost for eval difference description matching: ${cost:.4f}")
    matches = [b[0] for b in res]

    ## recover predictions
    predictions = []  # matched predictions
    for i, (row, differences_gt, pred_unmatched, match) in enumerate(
            zip(dataset, differences_gt_all, predictions_unmatched, matches)):

        # init the stuff to populate
        log_description_match = []
        pred = {}

        # verify that the `matching` output obeys some basic matching properties
        match = _verify_matching_properties(match, differences_gt,
                                            pred_unmatched, i)

        # iterate over the matches
        pred = {}
        for k, v in match.items():
            pred[k] = {}
            pred[k]['gt_description'] = row['differences_annotated'][k][
                'description']
            pred[k]['pred_description'] = pred_unmatched.get(v, {}).get(
                'description', None)
            pred[k]['pred'] = pred_unmatched.get(v, {}).get('prediction', None)
            pred[k]['pred_key'] = v
            pred[k]['gt_key'] = k

        # save the content
        predictions.append(pred)

    return predictions


def add_extra_details(dataset, predictions):
    """ """
    for sample, pred in zip(dataset, predictions):

        differences_gt = sample['differences_gt']
        # add entries for the missing gt difference keys
        keys_gt = [
            key for key, value in differences_gt.items() if value is not None
        ]
        for key in keys_gt:
            if key not in pred:
                pred[key] = {'prediction': None, 'description': None}
        assert set(keys_gt) == set(pred.keys())

        # add the extra info that would have been added by the matching
        for k in pred.keys():
            pred[k]['pred'] = pred[k]['prediction']
            del pred[k]['prediction']
            # pred[k]['gt'] = differences_gt[k]
            pred[k]['gt_key'] = k
            pred[k]['pred_key'] = k
            pred[k]['is_opposite'] = 0
            pred[k]['pred_description'] = pred[key]['description']
            pred[k]['gt_description'] = sample['differences_annotated'][k][
                'description']

    return predictions


def test_reverse_statements(predictions, seed, batch_size=20):
    """ 
    Test if it's opposite. 
    If it is the opposite, then flip the prediction
    We use batch size because performance seems to get worse with too many 
    examples at once. 
    """

    statements = []
    idx = 0
    for pred in predictions:
        for k, v in sorted(pred.items(), key=lambda x: int(x[0])):
            # if no match, then skip
            if pred[k]['pred_description'] is None:
                continue
            statements.append(
                [pred[k]['gt_description'], pred[k]['pred_description']])
            idx += 1

    # make prompts, based on the batch size
    batch_prompts_text = []
    for i in range(0, len(statements), batch_size):
        prompt = prompt_open_eval_check_opposite
        prompt = prompt.replace("{statements}",
                                str(statements[i:i + batch_size]))
        batch_prompts_text.append(prompt)

    # run the prompts
    seeds = [seed for _ in range(len(batch_prompts_text))]
    res = openai_api.call_gpt_batch(batch_prompts_text,
                                    seeds=seeds,
                                    model="gpt-4o-mini")
    cost = sum([r[1] for r in res])
    logging.info(f"Cost for eval on 'is_opposite' statement: ${cost:.4f}")
    is_opposite = []
    for r in res:
        is_opposite += r[0]['results']
    if not all(val in {'0', '1'} for val in is_opposite):
        raise ValueError(f"LLM issue in `test_reverse_statements`. The LLM out " \
            "should be a list with valuesin {'0', '1'}."
            f"This might be fixed by (i) changing random seed, or (ii) changing "
            "lowering the batch_size, which controls how many statement pairs are "\
            "evaluated per LLM call")
    if len(is_opposite) != len(statements):
        raise ValueError(f"LLM issue in `test_reverse_statements`. Array has the wrong " \
            "size. Try lowering batch_size, or the random seed. "
            )

    # put the 'is_opposite' predictions back
    idx = 0
    for pred in predictions:
        for k, v in sorted(pred.items(), key=lambda x: int(x[0])):
            # case 1: there was no match
            if pred[k]['pred_description'] is None:
                pred[k]['is_opposite'] = None
                continue
            # otherwise there was a match
            is_op = is_opposite[idx]
            if is_op == '1':
                pred[k]['pred'] = flip_abc(pred[k]['pred'])
            pred[k]['is_opposite'] = is_op
            idx += 1
    assert idx == len(is_opposite)

    return predictions


def make_eval_df(dataset, predictions, results_dir=None):
    # add the gt to the predictions and sample key to the predictions objects
    for row, pred in zip(dataset, predictions):
        differences_gt = {
            k: v
            for k, v in row['differences_gt'].items() if v is not None
        }
        assert set(differences_gt.keys()) == set(pred.keys())
        for k in pred.keys():
            pred[k]['gt'] = differences_gt[k]
            pred[k]['sample_key'] = row['sample_key']

    # first flatten then put to dataframe
    flattened_data = []
    for item in predictions:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)

    # some col reordering for convenient printing
    first_cols = ['gt', 'pred', 'sample_key']
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]

    return df


def _verify_matching_properties(match, differences_gt, pred_unmatched,
                                row_idx):
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
    hallucinated_match_vals = set(match_vals) - set(pred_unmatched.keys())
    if len(hallucinated_match_vals):

        # raise ValueError(
        logging.warning(
            f"LLM issue, row {row_idx}. The matches dict: [{match}] has values that are not one of "\
            f" the keys in the predictions: {pred_unmatched.keys()}\nHallucinated: {hallucinated_match_vals}.\n "
            "Change the seed passed in to eval_viddiff() function and retry matching")

        for k in list(match.keys()):
            if match[k] in hallucinated_match_vals:
                match[k] = None

    # no 'gt difference; is duplicated

    return match


def flip_abc(r):
    if r in ('a', 'b'):
        return {'a': 'b', 'b': 'a'}[r]
    else:
        return r


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
Provide your response as a JSON object containing an array of "0" or "1" values:
{"results" : ["1", "0", "0", ...]}

Input:
The list of statement pairs to analyze will be provided in the following format:

{statements}
"""
