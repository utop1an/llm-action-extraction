
import json
import re
from .solver import Solver
from llm import generate_prompt, generate_responses

class GPT3ToPlan(Solver):

    def __init__(self, datasets, model_name = 'gpt-4o-mini'):
        self.model_name = model_name
        self.examples = self.generate_examples(datasets)
        
    def generate_examples(self, datasets):
        examples = {}
        for ds_name, dataset in datasets.items():
            scored = []
            
            for i, item in enumerate(dataset):
                acts = item['acts']
                op_ex_count = sum(1 for act in acts if act['act_type'] in [2, 3])
                proportion = op_ex_count / len(acts) if len(acts) > 0 else 0
                scored.append((proportion, i))
            scored.sort(reverse=True)
            top2_indices = [i for _, i in scored[:2]]
            top2 = [dataset[i] for i in top2_indices]

            example_text = ""
            for item in top2:
                paragraph = []
                sents = item['sents']
                for sent in sents:
                    sentence = " ".join(sent)
                    paragraph.append(sentence)
                paragraph_text = '.\n'.join(paragraph) + '.'
                example_text += f"TEXT:\n{paragraph_text}\n"
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
            examples[ds_name] = example_text
        return examples

    def solve(self, paragraph, ds_name=""):
        example = self.examples.get(ds_name, [])
        if not example:
            raise ValueError(f'No examples found for dataset: {ds_name}')
        prompt = generate_prompt('gpt3_to_plan',  {'nl': paragraph, 'egs': example})
        response = generate_responses(self.model_name, prompt, log=True)['content']
        
        acts = response.strip().split(';')
        results = []
        for item in acts:
            m = re.match(r'(\w+)\((.*?)\)', item.strip())
            if m:
                act_name = m.group(1)
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
