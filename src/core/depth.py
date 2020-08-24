### Import modules
import cv2,PIL
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from .utils import utils
from .base import BaseClass


class DepthEst(BaseClass):

    def __init__(self, name='Depth Estimator', device="cpu"):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = device.lower()

        #Create net
        self.predictor = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.predictor.to(self.device).eval()

        #Create transforms
        self.transform_ = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.transform_.default_transform


    def predict(self,image):
        """
        Does depth estimation on a single image. In order to perform batch classification,
        you can either call this predict() function in a for-loop or alternatively (advanced)
        try to modify this predict() function to perform batch-inferencing.

        Input:
            image: cv2 type object

        Output:
            predictions: torch.tensor object
        """

        #Cast PIL image to cv2 format if needed
        # if type(image) != np.ndarray:
        #     image = utils.pil_to_cv2(image)

        #Transform / preprocess as required by trained model
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image_).to(self.device)

        #Inference
        with torch.no_grad():
            output_ = self.predictor(image)     
            
            #Post process
            output = torch.nn.functional.interpolate(
                output_.unsqueeze(1),
                size=image_.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            output = output.cpu().numpy()

        return output

        


    def visualize(self,img,output,figsize=(10,10)):
        """
        Simple visualizing function for depth estimator output.

        Input:
            img: cv2 type object. Original raw image.
            outputs: torch.tensor object returned by the predict() function

        Output:
            None
        """


        #Plot image
        fig,ax = plt.subplots(2,figsize = figsize)

        ax[0].imshow(img[:,:,::-1], interpolation='none')
        ax[0].set_title('Original Image')   

        ax[1].imshow(output, interpolation='none')
        ax[1].set_title('Depth Map')   

        plt.show()



    def save_image(self,image,output_path,file_name="my_pic",file_fmt="jpg"):
        """
        Helper function to save image to disk as jpg file.
        """
        name = output_path + "/" + file_name + "." + file_fmt
        return cv2.imwrite(name, image)