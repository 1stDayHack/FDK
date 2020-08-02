### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class TextGen(BaseClass):

    def __init__(self, name='GPT-2 TextGen',max_length=12):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.device = 1 if torch.cuda.is_available() else -1

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2")
        self.predictor = pipeline('text-generation', 
                                  model = self.model,
                                  tokenizer = self.tokenizer
                                  device = self.device,
                                  max_length = self.max_length)



    def predict(self,text):
        """
        Does depth estimation on a single image. In order to perform batch classification,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            image: str object. Seed text to be used for generation.

        Output:
            predictions: list object. Generated text.
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



class TextGen_Large(TextGen):

    def __init__(self, name='GPT-2 Large TextGen',max_length=12):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.device = 1 if torch.cuda.is_available() else -1

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2-large")
        self.predictor = pipeline(self.lp, 
                                  model = self.model,
                                  tokenizer = self.tokenizer
                                  device = self.device,
                                  max_length = self.max_length)




class TextGen_XL(TextGen):

    def __init__(self, name='GPT-2 XL TextGen',max_length=12):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.device = 1 if torch.cuda.is_available() else -1

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-XL")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2-XL")
        self.predictor = pipeline(self.lp, 
                                  model = self.model,
                                  tokenizer = self.tokenizer
                                  device = self.device,
                                  max_length = self.max_length)



class TextGen_Lite(TextGen):

    def __init__(self, name='GPT-2 Lite TextGen',max_length=12):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.device = 1 if torch.cuda.is_available() else -1

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelWithLMHead.from_pretrained("distilgpt2")
        self.predictor = pipeline(self.lp, 
                                  model = self.model,
                                  tokenizer = self.tokenizer
                                  device = self.device,
                                  max_length = self.max_length)
