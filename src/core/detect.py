### Import modules
import detectron2
from base import BaseClass
from detectron2.utils.logger import setup_logger

import numpy as np
import os, json, cv2, torch
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



class Detector(BaseClass):

    def __init__(self):
        super().__init__(name='detectron2',model):
        
        #Init name and metadata
        self.name = name
        self.model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" #checkout model_zoo for all configs
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        #Init config file
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = self.device

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        self.predictor = DefaultPredictor(self.cfg)



    def predict(self,image):
        """
        Does inference with all supported mode of inference by Detectron2.

        Input:
            image: cv2.Image type object

        Output:
            predictions: torch.tensor object
        """
        outputs = self.predictor(image)
        
        return outputs
        


    def visualize(self,image,outputs):
        """
        Simple single plot visualizing function.

        Input:
            image: cv2.Image type object
            outputs: torch.tensor object returned by the predict() function

        Output:
            None
        """
        viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0])), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image())
        plt.show()

        return None


     