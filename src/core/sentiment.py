### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class SentimentAnalyzer(BaseClass):

    def __init__(self, name='Sentiment Analyzer', device="cpu"): #alt: distilbert-base-uncased-finetuned-sst-2-english 
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 1 if device.lower() == "cpu" else -1

        #Create net
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelWithLMHead.from_pretrained("gpt2")
        self.predictor = pipeline('sentiment-analysis', 
                                #   model = self.model,
                                #   tokenizer = self.tokenizer,
                                  device = self.device)



    def predict(self,text):
        """
        Does sentiment analysis on a given text. In order to perform batch classification,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            text: str object. Seed text to be used for generation.

        Output:
            predictions: list object. Generated text.
        """


        #Infer
        output = self.predictor(text)

        return output

        


    def visualize(self,raw,output):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            raw: str object; default text
            output: str object; obj returned by predict()

        Output:
            None
        """

        #Print!
        for idx,i in enumerate(raw):
            pprint({"Raw Text":i,
                    "Sentiment":output[idx]['label'],
                    "Confidence":output[idx]['score']})

