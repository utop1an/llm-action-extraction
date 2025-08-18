from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMClient(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> Dict[str, Any]:
        pass

