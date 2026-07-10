
import json
import re
from .solver import Solver
from src.llm import generate_prompt, generate_responses

class GPT3ToPlan(Solver):

    prompt_name = "gpt3_to_plan"

    def __init__(self, datasets, model_name = 'gpt-4o-mini'):
        self.model_name = model_name
        self.datasets = datasets
        self.ranked_example_indices = self.rank_examples(datasets)
        self.examples = self.generate_examples(datasets)

    def rank_examples(self, datasets):
        ranked = {}
        for ds_name, dataset in datasets.items():
            scored = []

            for i, item in enumerate(dataset):
                acts = item['acts']
                op_ex_count = sum(1 for act in acts if act['act_type'] in [2, 3])
                proportion = op_ex_count / len(acts) if len(acts) > 0 else 0
                scored.append((proportion, i))
            scored.sort(reverse=True)
            ranked[ds_name] = [i for _, i in scored]
        return ranked

    def format_example(self, item):
        paragraph = []
        sents = item['sents']
        for sent in sents:
            sentence = " ".join(sent)
            paragraph.append(sentence)
        paragraph_text = '.\n'.join(paragraph) + '.'
        example_text = f"TEXT:\n{paragraph_text}\n"
        acts = item['acts']
        words = item['words']
        _acts = []
        for act in acts:
            act_idx = act['act_idx']
            act_name = words[act_idx]
            obj_idxs = act['obj_idxs'][0]
            obj_names = [words[ind] for ind in obj_idxs]
            act_text = f"{act_name}({','.join(obj_names)})"
            _acts.append(act_text)
        acts_text = ';'.join(_acts)
        example_text += f"ACTIONS:\n{acts_text}\n"
        return example_text

    def generate_examples(self, datasets, exclude_doc_ids=None):
        exclude_doc_ids = exclude_doc_ids or {}
        examples = {}
        for ds_name, dataset in datasets.items():
            excluded = exclude_doc_ids.get(ds_name)
            top2_indices = [
                i for i in self.ranked_example_indices.get(ds_name, [])
                if i != excluded
            ][:2]
            example_text = ""
            for i in top2_indices:
                example_text += self.format_example(dataset[i])
            examples[ds_name] = example_text
        return examples

    def build_prompt(self, paragraph, ds_name="", doc_id=None):
        if ds_name not in self.datasets:
            raise ValueError(f'No examples found for dataset: {ds_name}')
        if doc_id is None:
            example = self.examples.get(ds_name, "")
        else:
            example = self.generate_examples(
                self.datasets,
                exclude_doc_ids={ds_name: doc_id},
            ).get(ds_name, "")
        return generate_prompt(self.prompt_name, {'nl': paragraph, 'egs': example})

    def parse_json(self, string):
        acts = string.strip().split(';')
        results = []
        for item in acts:
            m = re.match(r'\s*([^()]+?)\s*\((.*)\)\s*$', item.strip())
            if m:
                act_name = m.group(1).strip()
                arguments = m.group(2).split(',') if m.group(2) else []
            else:
                act_name = item.strip()
                arguments = []
            act_obj = {
                'verb': act_name,
                'arguments': arguments
            }
            results.append(act_obj)
        return results

    def solve(self, paragraph, ds_name="", doc_id=None):
        prompt = self.build_prompt(paragraph, ds_name=ds_name, doc_id=doc_id)
        response = generate_responses(self.model_name, prompt, log=True)['content']
        return self.parse_json(response)
