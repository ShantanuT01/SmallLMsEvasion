from abc import ABC, abstractmethod
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline, Pipeline

from llm_attacker.constants import *


class BlackBoxDetector:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, ai_label: str) -> None:
        self.__model = model
        self.__tokenizer = tokenizer
        self.__pipeline = pipeline("text-classifcation", model=self.__model, tokenizer=self.__tokenizer)
        self.__ai_label = ai_label


    def evaluate(self, text: str) -> float:
        scores = self.__pipeline(text, return_all_scores=True)
        for score in scores[0]:
            if score[LABEL] == self.__ai_label:
                return score[SCORE]
        raise Exception(f"{self.__ai_label} not found!")

    

class Attacker(ABC):
    def __init__(self, pipe: Pipeline, detector: BlackBoxDetector):
        self.__pipe = pipe
        self.__detector = detector


    @abstractmethod
    def attack(self, text: str):
        pass



class SynonymAttacker(Attacker):

    def __init__(self, pipe: Pipeline, detector: BlackBoxDetector):
        super().__init__(pipe, detector)


    def attack(self, text):
        pass



class ParaphraseAttacker(Attacker):

    def __init__(self, pipe: Pipeline, detector: BlackBoxDetector):
        super().__init__(pipe, detector)

    def attack(self, text):
        pass


    