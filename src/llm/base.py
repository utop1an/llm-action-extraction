from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMClient(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        pass

