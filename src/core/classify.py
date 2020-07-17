### Import modules
from base import BaseClass
from utils.mouse import MouseSelector
from utils.variables import imagenet_map,imagenet_stats

import torchvision.transforms as transforms
import numpy as np
import os, json, cv2, torch, torchvision, PIL
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class Classifier(BaseClass):

    def __init__(self, model, name='ImageNet_Classifier'):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        #Create net
        self.predictor = torchvision.models.wide_resnet101_2(pretrained=True)
        self.predictor.to(self.device).eval()

        #Init helper
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(imagenet_stats['mean'],
                                                                  imagenet_stats['std'])])


    def predict(self,image):
        """
        Does classification on a single image. In order to perform batch classification,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            image: cv2 type object

        Output:
            predictions: torch.tensor object
        """

        #Cast cv2 image to PIL format if needed
        if type(image) != PIL.Image:
            cv2_im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(cv2_im)

        #Transform / preprocess as required by trained model
        images_tf = self.transform(image).unsqueeze(0) #make batch dimension
        
        #Predict / Inference
        output = self.predictor(images_tf).squeeze(0) #remove batch dimension

        return output
        


    def visualize(self,image,output):
        """
        Simple single plot visualizing function.

        Input:
            image: cv2 type object
            outputs: torch.tensor object returned by the predict() function

        Output:
            None
        """

        #Check
        assert len(output.shape) <= 3, "Error! The visualize() function only accept individual image only, NOT batches."

        #Cast cv2 image to PIL format if needed
        if type(image) != PIL.Image:
            cv2_im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(cv2_im)

        #Plot image
        plt.imshow(image)
        plt.show()

        #Print labels
        print("This is a(n) {}".format(imagenet_map[output]))
     