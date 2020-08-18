### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class Summarizer(BaseClass):

    def __init__(self, name='BART Summarizer',max_length=130,min_length=30):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 1 if torch.cuda.is_available() else -1
        self.max_length = max_length
        self.min_length = min_length

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelWithLMHead.from_pretrained("facebook/bart-large-cnn")
        self.predictor = pipeline('summarization', 
                                 model = self.model,
                                 tokenizer = self.tokenizer,
                                 device = self.device)



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
        output = self.predictor(text,max_length=self.max_length,min_length=self.min_length)

        return output

        


    def visualize(self,raw,output):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            img: str object

        Output:
            None
        """

        #Print!
        pprint({"Raw":raw,
                "Summarized":output[0]['summary_text']})

