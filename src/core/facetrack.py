### Import modules
from .base import BaseClass
from .base_libs.BlazeFace.blazeface import BlazeFace

import numpy as np
import os, json, cv2, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class FaceTracker(BaseClass):


    def __init__(self, name='BlazeFace', device="cpu"):
        super().__init__(name)
        
        #Check for weight files first
        if not os.path.isfile("src/core/base_libs/BlazeFace/blazeface.pth"):
            raise Exception("Warning! Missing weight file for model. Download it at https://drive.google.com/drive/folders/1HzUseRlhoYluTOEnS3oRpoFJ_gxXeubV?usp=sharing")

        #Init name and metadata
        self.name = name
        # self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.device = device.lower()
        self.weight_path = 'src/core/base_libs/BlazeFace/blazeface.pth' #replace me
        self.anchor_path = 'src/core/base_libs/BlazeFace/anchors.npy' #replace me

        #Create net
        self.predictor = BlazeFace().to(self.device)
        self.predictor.load_weights(self.weight_path)
        self.predictor.load_anchors(self.anchor_path)

        #Optionally change the thresholds:
        self.predictor.min_score_thresh = 0.75
        self.predictor.min_suppression_threshold = 0.3

        #Temp message
        print("Works only on image size of 128x128. For larger images, perform rescaling input and output.")


    def predict(self,image):
        """
        Performs face detection and facial keypoint detection.
        Unfortunately only works for low resolution input for now which is not idea.
        Recommended to be used for close ups only.
 
        Input:
            image: cv2 type object

        Output:
            predictions: torch.tensor object
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Cast as right format
        outputs = self.predictor.predict_on_image(image)
        
        return outputs
        


    def visualize(self,image,outputs,with_keypoints=True):
        """
        Simple single plot visualizing function.

        Input:
            image: cv2 type object
            outputs: torch.tensor object returned by the predict() function
            with_keypoints: boolean; if True, will return visualization with keypoints drawn on

        Output:
            None
        """
        #Cast as right format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.grid(False)
        ax.imshow(image)
        
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()

        if outputs.ndim == 1:
            outputs = np.expand_dims(outputs, axis=0)

        print("Found %d faces" % outputs.shape[0])
            
        for i in range(outputs.shape[0]):
            ymin = outputs[i, 0] * image.shape[0]
            xmin = outputs[i, 1] * image.shape[1]
            ymax = outputs[i, 2] * image.shape[0]
            xmax = outputs[i, 3] * image.shape[1]

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=1, edgecolor="r", facecolor="none", 
                                    alpha=outputs[i, 16])
            ax.add_patch(rect)

            if with_keypoints:
                for k in range(6):
                    kp_x = outputs[i, 4 + k*2    ] * image.shape[1]
                    kp_y = outputs[i, 4 + k*2 + 1] * image.shape[0]
                    circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                            edgecolor="lightskyblue", facecolor="none", 
                                            alpha=outputs[i, 16])
                    ax.add_patch(circle)
            
        plt.show()

        return None


     