### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class QuesAns(BaseClass):

    def __init__(self, name='BERT QuesAns', device="cpu"):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 1 if device.lower() == "cpu" else -1

        #Create net. Commented out; seems like default works, other models/tokenizer will have errors.
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        # self.model = AutoModelWithLMHead.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.predictor = pipeline('question-answering',
                                  device = self.device)



    def predict(self,text):
        """
        Does Q&A on given text. In order to perform batch inference,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            text: str object. Seed text to be used for generation.

        Output:
            predictions: list object. Generated text.
        """


        #Infer
        output = self.predictor(question=text['question'],
                                context=text['context'])

        return output

        


    def visualize(self,text):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            text: str object; obj returned by predict()

        Output:
            None
        """

        #Print!
        pprint(text)

