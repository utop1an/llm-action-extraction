from config import TASKS
from .subtask import SubTask

class Task:
    def __init__(self, name, description, prompt, parameters):
        self.name = self.name
        conf = TASKS.get(name)
        if not conf:
            raise ValueError(f"Task {name} not found in config: {TASKS.keys()}")
        self.description = conf["description"]
        self.prompt = conf["prompt"]
        self.parameters = parameters
        self.subtasks = []

    def __repr__(self):
        return f"Task({self.name}: {self.parameters})"
    
    def add_subtask(self, subtask):
        if not isinstance(subtask, SubTask):
            raise TypeError("subtask must be an instance of SubTask")
        self.subtasks.append(subtask)

    def get_subtask(self, name):
        for subtask in self.subtasks:
            if subtask.name == name:
                return subtask
        raise ValueError(f"SubTask {name} not found in Task {self.name}")
        
    def __getitem__(self, name):
        return self.get_subtask(name)

    def __itter__(self):
        return iter(self.subtasks)

    