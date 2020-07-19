### Import modules
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from base_libs.ESRGAN import RRDBNet_arch as arch
from base import BaseClass


class SuperReser(BaseClass):

    def __init__(self, model, name='Super Resolution ESRGAN'):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        #Create net
        self.weight = 'baselibs/ESRGAN/models/RRDB_ESRGAN_x4.pth' 
        self.predictor = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.predictor.load_state_dict(torch.load(self.weight), strict=True)
        self.predictor.to(self.device).eval()


        #Init helper
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0,0,0],
                                                                  [255,255,255])])


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
        images_tf = self.transform(image).float().unsqueeze(0) #make batch dimension
        
        #Predict / Inference
        output = self.predictor(images_tf).data.squeeze(0).float().cpu().clamp_(0,1).numpy() #remove batch dimension

        #Post process according to model's construction
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

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
     