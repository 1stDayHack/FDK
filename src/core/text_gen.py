### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class _TextGen(BaseClass):

    def __init__(self, name='GPT-2 TextGen Base Class', device="cpu"):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 1 if device.lower() == "cpu" else -1



    def predict(self,text):
        """
        Does text generation based on given seed text. In order to perform batch inference,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            text: str object. Seed text to be used for generation.

        Output:
            predictions: list object. Generated text.
        """


        #Infer
        output = self.predictor(text,
                                max_length=self.max_length,
                                num_return_sequences=self.num_return_sequences)

        return output

        


    def visualize(self,text):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            text: str object

        Output:
            None
        """

        #Print!
        pprint(text)



class TextGen_Base(_TextGen):

    def __init__(self, name='GPT-2 TextGen',max_length=12,num_return_sequences=3):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.num_return_sequences=num_return_sequences

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2")
        self.predictor = pipeline('text-generation', 
                                  model = self.model,
                                  tokenizer = self.tokenizer,
                                  device = self.device)



class TextGen_Large(_TextGen):

    def __init__(self, name='GPT-2 Large TextGen',max_length=12,num_return_sequences=3):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.device = 1 if torch.cuda.is_available() else -1
        self.num_return_sequences=num_return_sequences


        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2-large")
        self.predictor = pipeline('text-generation', 
                                  model = self.model,
                                  tokenizer = self.tokenizer,
                                  device = self.device)




class TextGen_XL(_TextGen):

    def __init__(self, name='GPT-2 XL TextGen',max_length=12,num_return_sequences=3):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.num_return_sequences=num_return_sequences
        

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-XL")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2-XL")
        self.predictor = pipeline('text-generation', 
                                  model = self.model,
                                  tokenizer = self.tokenizer,
                                  device = self.device)



class TextGen_Lite(_TextGen):

    def __init__(self, name='GPT-2 Lite TextGen',max_length=12,num_return_sequences=3):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.max_length = max_length
        self.num_return_sequences=num_return_sequences

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelWithLMHead.from_pretrained("distilgpt2")
        self.predictor = pipeline('text-generation', 
                                  model = self.model,
                                  tokenizer = self.tokenizer,
                                  device = self.device)
