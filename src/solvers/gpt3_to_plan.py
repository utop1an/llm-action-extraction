
import json
import re
from .solver import Solver
from llm import generate_prompt, generate_responses

class GPT3ToPlan(Solver):

    def __init__(self, model_name = 'gpt-4o-mini'):
        self.model_name = model_name
        

    def solve(self, paragraph):
        prompt = generate_prompt('gpt3_to_plan',  {'nl': paragraph})
        response = generate_responses(self.model_name, prompt, log=True)['content']
        m = re.search(r"```(?:json|jsonc)?\s*([\s\S]*?)\s*```", response, re.I)
        payload = m.group(1).strip() if m else response
        obj = json.loads(payload)
        return obj
