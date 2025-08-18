from config import SUBTASKS


class SubTask:
    def __init__(self, task, name, description, prompt, parameters):
        self.task = task
        self.name = name
        conf = SUBTASKS.get(name)
        if not conf:
            raise ValueError(f"SubTask {name} not found in config: {SUBTASKS.keys()}")
        self.description = conf["description"]
        self.prompt = conf["prompt"]
        self.parameters = parameters

    def __repr__(self):
        return f"SubTask({self.task.name}-{self.name}: {self.parameters})"