from abc import ABC, abstractmethod

class Solver(ABC):
    @abstractmethod
    def solve(self, paragraph, ds_name=""):
        raise NotImplementedError