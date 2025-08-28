from ..config import PROMPTS, generate_prompt


class SubTask:
    def __init__(self, name):
        self.name = name
        conf = PROMPTS.get(name)
        if not conf:
            raise ValueError(f"SubTask {name} not found in config: {PROMPTS.keys()}")
        self.description = conf["description"]
        self.prompt = conf["prompt"]
        self.parameters = conf["parameters"]

    def __repr__(self):
        return f"SubTask({self.task.name}-{self.name}: {self.parameters})"
    
    def get_prompt(self, parameters):
        # check parameter keys
        return generate_prompt(self.name, parameters)

    def get_response(self, parameters, model, is_async=False):
        from ..chat_completion import generate_responses
        prompt = self.get_prompt(parameters)

        return generate_responses(model,prompt,is_async)