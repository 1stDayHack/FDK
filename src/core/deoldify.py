### Import modules
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import fastai
import urllib.request

from pathlib import Path
from PIL import Image 
from pprint import pprint
from .base_libs.deoldify.deoldify.visualize import *
from .base_libs.deoldify.deoldify._device import _Device
from .base_libs.deoldify.deoldify.device_id import DeviceId
from .utils import utils
from .base import BaseClass


class Deoldifier(BaseClass):

    def __init__(self, name='Deoldifier-Stable', render_factor=35, device="cpu"): 
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = DeviceId.CPU if device.lower() == "cpu" else DeviceId.GPU0
        device_ = _Device()
        device_.set(device=self.device)

        #Check if weight exist
        self.root_path = 'src/core/base_libs/deoldify/'
        self.weight_path = self.root_path + 'models/'
        self.weight_name = 'ColorizeStable_gen.pth'

        if not os.path.isdir(self.weight_path):
            os.mkdir(self.weight_path)
        
        if self.weight_name not in os.listdir(self.weight_path):
            print('Fetching deoldify weights...')
            url = 'https://www.dropbox.com/s/mwjep3vyqk5mkjc/ColorizeStable_gen.pth?dl=0'
            urllib.request.urlretrieve(url, self.weight_path + self.weight_name)


        #Create net and set hyperparameters
        self.predictor = get_image_colorizer_(path_=Path(self.root_path), artistic=False)
        self.render_factor = render_factor
        self.watermarked = True



    def predict(self,path_):
        """
        Does depth estimation on a single image. In order to perform batch classification,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            image: str object. Path to raw image.

        Output:
            predictions: PIL objs. Raw image and Colorized image.
        """

        #Infer
        output_pth = self.predictor.plot_transformed_image(path_, render_factor=self.render_factor, compare=True, watermarked=self.watermarked)
        
        #Read images to return
        output = Image.open(output_pth)
        input_ = Image.open(path_)

        return input_,output

        


    def visualize(self,raw,colorized,figsize=(20,20)):
        """
        Simple function to call pretty-print for a neater text representation.

        Input:
            img: str object

        Output:
            None
        """

        #Viz piggybacking on deoldify's visualizing function
        _plot_comparison(
                figsize, self.render_factor, False, raw, colorized
            )

