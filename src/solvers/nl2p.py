from .solver import Solver
from llm import generate_prompt, generate_responses
import json
import re

class NL2P(Solver):

    def __init__(self, model_name):
        self.model_name = model_name

    def solve(self, paragraph):
        return self.get_verb_args(paragraph)
        
    

    def get_verb_args(self, paragraph):
        prompt = generate_prompt('verb_arg', {'nl': paragraph})
        response = generate_responses(self.model_name, prompt, log=True)['content']
        m = re.search(r"```(?:json|jsonc)?\s*([\s\S]*?)\s*```", response, re.I)
        payload = m.group(1).strip() if m else response
        obj = json.loads(payload)
        return obj
