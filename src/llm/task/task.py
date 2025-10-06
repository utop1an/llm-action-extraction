from ..config import PROMPTS, TASK_FUNCTIONS, generate_prompt
from .subtask import SubTask
from typing import List, Union

class Task:
    def __init__(self, name:str, subtasks: List[Union[str, SubTask]]=[]):
        self.name = name
        conf = PROMPTS.get(name)
        if not conf:
            raise ValueError(f"Task {name} not found in config: {PROMPTS.keys()}")
        self.description = conf["description"]
        self.prompt = conf["template"]
        self.parameters = conf["parameters"]
        func_conf = TASK_FUNCTIONS.get(name)
        self.func = func_conf["function"] if func_conf else None
        self.subtasks = [self.add_subtask(subtask) for subtask in subtasks]

    def __repr__(self):
        return f"Task({self.name}: {self.parameters})"
    
    def __str__(self):
        return self.prompt.format(**self.parameters)

    def add_subtask(self, subtask: Union[str, SubTask]):
        if isinstance(subtask, SubTask):
            self.subtasks.append(subtask)
        elif isinstance(subtask, str):
            self.subtasks.append(SubTask(subtask))
        else:
            raise TypeError("subtask must be an instance of SubTask or str (subtask name)")


    def get_subtask(self, name):
        for subtask in self.subtasks:
            if subtask.name == name:
                return subtask
        raise ValueError(f"SubTask {name} not found in Task {self.name}")
        
    def __getitem__(self, name):
        return self.get_subtask(name)

    def __itter__(self):
        return iter(self.subtasks)
    
    def get_prompt(self, parameters):
        # check parameter keys
        return generate_prompt(self.name, parameters)

    def get_llm_response(self, parameters, model, is_async=False):
        from ..chat_completion import generate_responses
        prompt = self.get_prompt(parameters)

        return generate_responses(model,prompt,is_async)

    def test_call(self, parameters, model, is_async=False):
        return True
        
    def solve_task(self, parameters, model, is_async=False):
        if not self.func:
            return self.get_llm_response(parameters, model, is_async)
        else:
            return self.func(self, parameters, model, is_async)
    


    



    