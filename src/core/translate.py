### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class Translator(BaseClass):

    def __init__(self, name='T5 Translator',language_pairs='translation_en_to_de'):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.lp = language_pairs
        self.device = 1 if torch.cuda.is_available() else -1

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.model = AutoModelWithLMHead.from_pretrained("t5-base")
        self.predictor = pipeline(self.lp, 
                                  model = self.model,
                                  tokenizer = self.tokenizer
                                  device = self.device)



    def predict(self,text):
        """
        Does depth estimation on a single image. In order to perform batch classification,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            image: str object. Text to be translated.

        Output:
            predictions: list object. Translated text.
        """


        #Infer
        output = self.predictor(text)

        return output

        


    def visualize(self,text):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            img: str object

        Output:
            None
        """

        #Print!
        pprint(text)

