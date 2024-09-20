"""
Take an activity description and use an LLM to propose differences and action stages
"""
import ipdb
import numpy as np
import tqdm
import sys
import json
import logging
import copy
from pathlib import Path
from collections import OrderedDict
import itertools

from viddiff_method import prompts
from apis import openai_api
from proposer_types import Difference, Stage, Proposal, CustomJsonEncoder
import eval_viddiff


class Proposer():
    """
    Create self.proposals, which is a dict from the 'sample_key' of each sample
    to a Proposer object, which has:
    - proposed differences
    - sub-action stages
    - each difference is associated to one or more stages
    """

    def __init__(self, args, args_logging, dataset, dataset_eval=None):
        """ 
        """
        # save configs
        self.args = args
        self.verbose = self.args.verbose

        # save dataset, and make sure samples are ordered
        self.dataset = dataset

        # logging subdirectory
        self.results_subdir = Path(args_logging.results_dir) / "proposer"
        self.results_subdir.mkdir(exist_ok=True, parents=True)

    def generate_proposals(self):

        # construct candidate differences. If eval_mode>0, read from gt
        if self.args.eval_mode == 0:
            self.query_1_differences()
        else:
            self.query_1_differences_gt()

        # get text retrieval information for action stages
        if self.args.eval_mode in (0, 1):
            self.query_2_subactions()  # actually has query 2 and query 3
        else:
            raise

        # link the differences to stages
        self.query_4_linking()

        if self.args.do_eval and self.args.eval_mode == 0:
            self.match_differences()

        return self.proposals

    def query_1_differences_gt(self):
        raise

    def query_1_differences(self):
        """
        If open eval mode, make one 'differece' proposal per sample. 
        We could have done one call per unique action, but we do one per sample 
        instead. 

        That's because, if you did one-per action, and you failed to predict a gt
        difference, then ALL the samples in that action automatically have an error.

        This way, the variance in accuracy between runs is reduced. 

        Each sample uses it's `sample_hash` attribute as part of its seed passed 
        to the LLM, added to self.args.seed
        """
        # get and verify template
        template_differences = prompts.lookup_prompts_proposer_1_differences[
            self.args.prompt_key_1_differences]
        if self.args.n_differences:
            assert "{n_differences}" in template_differences

        # one query per sample
        batch_texts = []
        batch_seeds = []
        sample_keys = []
        for sample in self.dataset:
            seed = self.args.seed + sample['sample_hash']
            prompt = template_differences.replace("{action}",
                                                  sample['action_description'])
            prompt = prompt.replace("{n_differences}",
                                    str(self.args.n_differences))

            batch_texts.append(prompt)
            batch_seeds.append(seed)
            sample_keys.append(sample['sample_key'])

        # call gpt for differences
        llm_batch = openai_api.call_gpt_batch(batch_texts,
                                              model=self.args.model,
                                              seeds=batch_seeds)

        cost = sum([b[1] for b in llm_batch])
        logging.info(f"Cost for difference generation: ${cost:.4f}")
        responses = [b[0] for b in llm_batch]

        # enforce max n_differences
        for res in responses:
            if len(res) > self.args.n_differences:
                res = dict(list(res.items())[:n_differences])
                logging.warning(f"A proposal had [{len(res)}'' differences " \
                f"but max allowed is {n_differences}")

        # log results to object and to file
        self.responses_1_differences = dict(zip(sample_keys, responses))
        self.results_subdir_differences = self.results_subdir / "1_differences"
        self.results_subdir_differences.mkdir(exist_ok=True, parents=True)
        for sample, res in zip(self.dataset, responses):
            f_save = self.results_subdir_differences / f"sample_{sample['sample_key']}_action_{sample['action']}.json"
            with open(f_save, 'w') as f:
                json.dump(res, f, indent=4)

    def query_2_subactions(self):
        """ 
        Query one action at a time.
        """
        template_subactions = prompts.lookup_prompts_proposer_2_subactions[
            self.args.prompt_key_2_subactions]
        # action_descriptions = list(
        #     self.dataset['lookup_idx_to_actions'].values())

        # get prompts
        batch_texts = []
        batch_seeds = []
        sample_keys = []
        action_descriptions = []
        for sample in self.dataset:
            seed = self.args.seed + sample['sample_hash']
            prompt_subactions = template_subactions.replace(
                "{action}", sample['action_description'])
            prompt_subactions = prompt_subactions.replace(
                "{n_retrieval_keys}", str(self.args.n_retrieval_keys))

            action_descriptions.append(sample['action_description'])
            sample_keys.append(sample['sample_key'])
            batch_texts.append(prompt_subactions)
            batch_seeds.append(seed)

        # run llm
        llm_batch = openai_api.call_gpt_batch(batch_texts,
                                              seeds=batch_seeds,
                                              # overwrite_cache=True,
                                              model=self.args.model)
        cost = sum([b[1] for b in llm_batch])
        responses = [b[0] for b in llm_batch]
        logging.info(f"Cost for stages generation: ${cost:.4f}")

        # log
        self.results_subdir_subactions = self.results_subdir / "2_subactions"
        self.results_subdir_subactions.mkdir(exist_ok=True, parents=True)

        self.responses_2_stages_ = dict(zip(sample_keys, responses))
        for sample, res in zip(self.dataset, responses):
            f_out = self.results_subdir_subactions / f"sample_{sample['sample_key']}_action_{sample['action']}_subactions_first.json"
            with open(f_out, 'w') as f:
                json.dump(res, f, indent=4)

        # optionally filter bad retrieval keys
        if self.args.filter_retrieval_keys:
            # prompts
            template_subactions_refine = prompts.lookup_prompts_proposer_2_subactions_refiner[
                self.args.prompt_key_3_subaction_filtering]
            batch_texts = []
            for sample_key, sample in zip(sample_keys, self.dataset):

                response_old = self.responses_2_stages_[sample_key]
                prompt_refine = template_subactions_refine.replace(
                    "{action}", sample['action_description'])
                prompt_refine = prompt_refine.replace(
                    "{stages}", json.dumps(response_old, indent=4))
                batch_texts.append(prompt_refine)

            llm_batch = openai_api.call_gpt_batch(batch_texts,
                                                  model=self.args.model,
                                                  seed=self.args.seed)
            cost = sum([b[1] for b in llm_batch])
            logging.info(f"Cost for retrieval key filtering: ${cost:.4f}")
            responses = [b[0] for b in llm_batch]

            # log
            self.responses_2_stages = dict(zip(sample_keys, responses))
            for sample, res in zip(self.dataset, responses):
                f_out = self.results_subdir_subactions / f"sample_{sample['sample_key']}_action_{sample['action']}_subactions_refined.json"
                with open(f_out, 'w') as f:
                    json.dump(res, f, indent=4)
        else:
            # if not doing the filtering
            self.responses_2_stages = self.responses_2_actionkey_to_stages_

    def query_4_linking(self):
        """ """

        template_linking = prompts.lookup_prompts_proposer_3_linking[
            self.args.prompt_key_4_linking]

        batch_texts = []
        for sample in self.dataset:
            sample_key = sample['sample_key']

            stages = self.responses_2_stages[sample_key]
            differences = self.responses_1_differences[sample_key]

            prompt_linking = template_linking.replace(
                "{action}", sample['action_description'])
            prompt_linking = prompt_linking.replace(
                "{stages}", json.dumps(stages, indent=4))
            prompt_linking = prompt_linking.replace(
                "{differences}", json.dumps(differences, indent=4))

            batch_texts.append(prompt_linking)

        # call llm
        llm_batch = openai_api.call_gpt_batch(batch_texts,
                                              model=self.args.model,
                                              # overwrite_cache=True,
                                              seed=self.args.seed)
        cost = sum([b[1] for b in llm_batch])
        logging.info(f"Cost for linking generation: ${cost:.4f}")
        responses = [b[0] for b in llm_batch]

        sample_keys = self.dataset['sample_key']
        self.responses_3_linking = dict(zip(sample_keys, responses))

        # log vlm response
        self.results_subdir_linking = self.results_subdir / "4_linking"
        self.results_subdir_linking.mkdir(exist_ok=True, parents=True)
        for sample_key, res in self.responses_3_linking.items():
            f_out = self.results_subdir_linking / f"sample_{sample_key}_linking.json"
            with open(f_out, 'w') as f:
                json.dump(res, f, indent=4)

        # construct the final Proposer object
        self.proposals = {}
        for sample in self.dataset:
            sample_key = sample['sample_key']
            differences = self.responses_1_differences[sample_key]
            stages = self.responses_2_stages[sample_key]
            links = self.responses_3_linking[sample_key]

            # some basic validation checks
            stage_names = [d['name'] for d in stages['stages']]
            difference_names = [d['name'] for d in differences.values()]

            linked_differences_names = sum(links.values(), [])
            if set(linked_differences_names) != set(difference_names):
                logging.warning(f"missing some differences in the linking")
                # raise ValueError(f"missing some differences in the linking")

            if not all(s in stage_names for s in links.keys()):
                logging.warning(
                    f"llm response has bad stage key, stage links \n real: {stage_names} \n generated: {links.keys()}"
                )

            for stage in stages['stages']:
                differences_linked = links[stage['name']]
                stage['differences'] = differences_linked

                if not all([p in difference_names
                            for p in differences_linked]):
                    raise ValueError(
                        f"llm response has bad difference name, differences \n real: {difference_names} \n generated: {differences_linked}"
                    )

            # some basic type checking. If this fails, then change random seed, try again,
            stages = [Stage(**stage) for stage in stages['stages']]
            differences = {
                k: Difference(**var)
                for k, var in differences.items()
            }
            proposal = Proposal(
                action_key=sample['action'],
                action_description=sample['action_description'],
                stages=stages,
                differences=differences)
            proposal.postprocess()
            self.proposals[sample['sample_key']] = proposal

    def match_differences(self):
        """ 
        In eval mode, find the correspondence between proposed and gt variations. 
        If self.args.drop_unmatched_diffs=True (and it is by default) then drop 
        any variations that had no matches. This avoids computational expense 
        of stuff that won't be evaluated anyway. 

        Warning: if `self.args.drop_unmatched_diffs`, then unmatched difference 
        keys get removed. That's fine in this framework because from this point 
        on, no candidate variation affects any other. But if that changes, then 
        it would no longer be correct to delete it. E.g. maybe in the last stage 
        of the system, you want to make a query about multiple variations at 
        once. 
        """

        # we need to have the field 'description' and that's it
        proposals_for_matching = []
        for sample in self.dataset:
            differences = self.proposals[sample['sample_key']].differences
            diff_set = {
                k: {
                    "description": v['description']
                }
                for k, v in differences.items()
            }
            proposals_for_matching.append(diff_set)

        matching = eval_viddiff.do_matching(self.dataset,
                                            proposals_for_matching,
                                            self.args.seed)

        def _clean_dict(d):
            keys_keep = ('pred_description', 'gt_description', 'pred_key')
            return {
                k:
                {key: value
                 for key, value in v.items() if key in keys_keep}
                for k, v in d.items()
            }
        matching = [_clean_dict(item) for item in matching]

        # filter out things that don't get matched
        self.results_subdir_matching = self.results_subdir / "5_matching"
        self.results_subdir_matching.mkdir(exist_ok=True, parents=True)
        with open(self.results_subdir_matching / "matching.json", 'w') as fp:
            json.dump(matching, fp, indent=4)

        # only keep the proposals that have a mapped gt value, and change the proposal key to the gt mapped key
        num_before = 0
        num_after = 0
        for i, sample in enumerate(self.dataset):
            differences = self.proposals[sample['sample_key']].differences
            num_before += len(differences)
            matches = matching[i]
            n_gt_diffs = len(matches)
            idx_mapping = {v['pred_key'] : k for k, v in matches.items() if v['pred_key'] != 'None'}
            n_recovered_diffs = len(idx_mapping)
            acc = n_recovered_diffs / n_gt_diffs

            # update the proposal 
            diffs_updated = {idx_mapping[k]: v for k, v in differences.items() if k in idx_mapping}
            self.proposals[sample['sample_key']].differences = diffs_updated
            num_after += len(diffs_updated)

        acc_kept = num_after / num_before
        logging.info(f"Recovered {acc_kept:.4f} of the things in the matching")

