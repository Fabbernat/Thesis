from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def ask(self, question: str) -> str:
        pass
