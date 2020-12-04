### Import modules
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import warnings

from imageio import imread
from PIL import Image
from pprint import pprint

from .base_libs.ImageCaptioner.caption import visualize_att, caption_image_beam_search
from .utils import utils
from .base import BaseClass


class ImageAnnotater(BaseClass):

    def __init__(self, name='Show_Attend_Tell Image Annotation', beam_size=5, device="cpu"):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = device.lower()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.beam_size = beam_size

        #Suppress annoying printouts | Dangerous :p
        warnings.filterwarnings("ignore")


        #Define paths
        self.base_pth = 'src/core/base_libs/ImageCaptioner/weights/'
        self.encoder_pth = self.base_pth + 'encoder.pth'
        self.decoder_pth = self.base_pth + 'decoder.pth'
        self.wordmap_pth = self.base_pth + 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
       

        #Create net, decoder and encoder.
        self.decoder = torch.load(self.decoder_pth)
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()

        self.encoder = torch.load(self.encoder_pth)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()


        #Load word map (word2ix)
        with open(self.wordmap_pth, 'r') as j:  self.word_map = json.load(j)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}  # ix2word



    def predict(self,image):
        """
        Does image annotation on a single image. In order to perform batch inference,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            image: PIL Image in array format.

        Output:
            predictions: list object. Generated text.
        """


        #Infer | Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(self.encoder, 
                                                self.decoder,
                                                image, 
                                                self.word_map, 
                                                self.beam_size)
        
        #Parse
        alphas = torch.FloatTensor(alphas)
        words = [self.rev_word_map[ind] for ind in seq]

        output = (words,alphas)

        return output


    def visualize(self,image,output):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            image: PIL object.
            output: tuple of (seq,alphas) from predict() function.

        Output:
            None
        """

        #Visualize 
        words,alphas = output
        visualize_att(image, words, alphas, True)





