### Import modules
import numpy as np
import torch
import matplotlib.pyplot as plt

from pprint import pprint
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
from .utils import utils
from .base import BaseClass


class Translator_M(BaseClass):

    """
    Tip, change the "task" argument to a different available model provided by MarianTransfomer to translate different language pairs!
    
    Manual langugage-pairs model download: http://opus.nlpl.eu/Opus-MT/ 
    Easy language-pair model download: https://huggingface.co/Helsinki-NLP 
    """

    def __init__(self, name='MarianTransformer Translator',task='Helsinki-NLP/opus-mt-en-ROMANCE'):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.task = task
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #Create net
        self.tokenizer = MarianTokenizer.from_pretrained(self.task)
        self.predictor = MarianMTModel.from_pretrained(self.task).to(self.device)



    def predict(self,text):
        """
        Does text translation on a given text. In order to perform batch inference,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            text: str object. Text to be translated.

        Output:
            predictions: list object. Translated text.
        """

        #Package
        if type(text) == 'str':
            text = [text] #Wrap

        #Infer and decode
        translated = self.predictor.generate(**self.tokenizer.prepare_translation_batch(text))
        translated_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return translated_text

        


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
        pprint({"Raw Text":raw,
                "Translation":output,
                "Task":self.task.split('/')[-1]})




class Translator_T5(BaseClass):
    """
    Tip, change the "task" argument to a different available model provided by MarianTransfomer to translate different language pairs!
    
    Find language-pair models at: https://huggingface.co/ 
    """

    def __init__(self, name='T5 Translator', task='translation_en_to_de', device="cpu"):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.task = task
        # self.device = 1 if torch.cuda.is_available() else -1
        self.device = -1 if device.lower() == "cpu" else 1

        #Create net
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.model = AutoModelWithLMHead.from_pretrained("t5-base")
        self.predictor = pipeline(self.task, 
                                  model = self.model,
                                  tokenizer = self.tokenizer,
                                  device = self.device)



    def predict(self,text):
        """
        Does textual translation on given text. In order to perform batch inference,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            text: str object. Text to be translated.

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
            text: str object; object returned by predict()

        Output:
            None
        """

        #Print!
        pprint(text)


