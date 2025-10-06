
from typing import Dict, List

class TypedObject:

    def __init__(self, obj_name: str, type_name: str = None):
        self.obj_name = obj_name
        self.type_name = type_name

    def __repr__(self):
        return f"{self.obj_name}?{self.type_name})"

class Verb:

    def __init__(self, verb: str, args: Dict[str, str]):
        self.verb = verb
        self.args = args

    def __repr__(self):
        return f"{self.verb}({self.args})"
    
class Action:

    def __init__(self, action_name, args: List[TypedObject]):
        self.action_name = action_name
        self.args = args

    def __repr__(self):
        return f"{self.action_name}({self.args})"


    