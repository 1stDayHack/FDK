### Import modules
import cv2,PIL,os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from .utils import utils
from .base_libs.ESRGAN import RRDBNet_arch as arch
from .base import BaseClass


class SuperReser(BaseClass):

    def __init__(self, name='Super Resolution ESRGAN', device="cpu"):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = device.lower()
        # self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        #Set weights and check
        self.weight_1 = 'src/core/base_libs/ESRGAN/models/RRDB_ESRGAN_x4.pth' 
        self.weight_2 = 'FDK/src/core/base_libs/ESRGAN/models/RRDB_ESRGAN_x4.pth' 
        self.weight = self.weight_1 if os.path.isfile(self.weight_1) else self.weight_2

        if not os.path.isfile(self.weight): 
            raise Exception("Model weight not found! Please download it at https://drive.google.com/file/d/1lq5UcxRWnZ5XUQzYD2lW2vFYBBFHdLff/view?usp=sharing")

        #Create net
        self.predictor = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.predictor.load_state_dict(torch.load(self.weight), strict=True)
        self.predictor.to(self.device).eval()


    def predict(self,image):
        """
        Does super-resolution upscaling on a single image. In order to perform batch inference,
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
        image = image * 1 / 255
        image = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        image = image.unsqueeze(0).to(self.device) #make batch dimension

        with torch.no_grad():
            output = self.predictor(image).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255).round()


        return output
        


    def visualize(self,original,output,figsize=(10,10)):
        """
        Simple visualizing function for pair of images.

        Input:
            image: cv2 type object
            outputs: torch.tensor object returned by the predict() function
            figsize: tuple; final visualization figure plot size

        Output:
            None
        """

        #Check
        assert len(output.shape) <= 3, "Error! The visualize() function only accept individual image only, NOT batches."
        
        #Plot image
        fig,ax = plt.subplots(2,figsize = figsize)

        ax[0].imshow(original[:,:,::-1], interpolation='none')
        ax[0].set_title('Original Image')   

        ax[1].imshow(output[:,:,::-1]/255, interpolation='none')
        ax[1].set_title('Super Resolution Result')   

        plt.show()


    def save_image(self,image,output_path,file_name="my_pic",file_fmt="jpg"):
        """
        Helper function to save image to disk as jpg file.
        """
        name = output_path + "/" + file_name + "." + file_fmt
        return cv2.imwrite(name, image)