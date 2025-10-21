from .solver import Solver
from llm import generate_prompt, generate_responses
import json
import re
import os

class NL2P(Solver):

    def __init__(self, model_name):
        self.model_name = model_name

    def log(self, step, result):
        log_dir = './logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = os.path.join(log_dir, 'nl2p_%s.jsonl' % step)
        with open(log_file_name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')
        

    def solve(self, paragraph, ds_name=""):
        verb_args = self.get_verb_args(paragraph)
        verb_args_filtered = self.remove_non_eventive_verbs(paragraph, verb_args)
        typed_verb_args = self.identify_verb_types(paragraph, verb_args_filtered)
        return typed_verb_args

    def parse_json(self, string):
        try:
            m = re.search(r"```(?:json|jsonc)?\s*([\s\S]*?)\s*```", string, re.I)
            payload = m.group(1).strip() if m else string
            obj = json.loads(payload)
            return obj
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            return None

    def get_verb_args(self, paragraph):
        prompt = generate_prompt('verb_args', {'nl': paragraph})
        response = generate_responses(self.model_name, prompt, temperature=0, log=True)['content']
        obj = self.parse_json(response)
        self.log('verb_args', json.dumps(obj))
        return obj
    
    def remove_non_eventive_verbs(self, paragraph, verb_args):
        prompt = generate_prompt('remove_non_eventive_verbs', {'nl': paragraph, 'verbs': json.dumps(verb_args)})
        response = generate_responses(self.model_name, prompt, temperature=0, log=True)['content']
        obj = self.parse_json(response)
        self.log('remove_non_eventive_verbs', json.dumps(obj))
        return obj

    def identify_verb_types(self, paragraph, verb_args):
        prompt = generate_prompt('identify_verb_types', {'nl': paragraph, 'verbs': json.dumps(verb_args)})
        response = generate_responses(self.model_name, prompt, temperature=0, log=True)['content']
        obj = self.parse_json(response)
        self.log('identify_verb_types', json.dumps(obj))
        return obj
    
