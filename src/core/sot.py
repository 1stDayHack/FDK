### Import modules
from base import BaseClass
from base_libs.DaSiamRPN.dasiamrpn.net import SiamRPNotb
from base_libs.DaSiamRPN.dasiamrpn.run_SiamRPN import SiamRPN_init, SiamRPN_track
from base_libs.DaSiamRPN.dasiamrpn.utils import rect_2_cxy_wh, cxy_wh_2_rect
from utils.mouse import MouseSelector

import numpy as np
import os, json, cv2, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class ObjectTracker(BaseClass):

    def __init__(self, model, name='DaSiamRPN'):
        super().__init__(name)
        
        #Init name and metadata
        self.name = name
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.weight_path = 'blazeface.pth' #replace me

        #Create net
        self.predictor = SiamRPNotb().to(self.device)
        self.predictor.load_state_dict(self.weight_path)
        self.predictor.eval().cuda()

        #Init helper
        self.MS = MouseSelector()

    def prime(self,image):
        """
        Initialize object tracker by annotating the target to be tracked on the 
        first frame of video or image sequence.
        
        Call this after instantiation and before any attempts to call predict().
        
        Input:
            X

        Output:
            Y
        """
        bbox_gt = self.init_bbox(image)
        target_pos, target_sz = rect_2_cxy_wh(bbox_gt)

        #Set state
        self.state = SiamRPN_init(image, target_pos, target_sz, self.predictor) 
        self.location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])



    def predict(self,image):
        """
        Does inference with all supported mode of inference by Detectron2.

        Input:
            image: cv2 type object

        Output:
            predictions: torch.tensor object
        """

        self.state = SiamRPN_track(self.state,image)
        self.location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])

        return self.location
        


    def visualize(self,image,outputs,with_keypoints=True):
        """
        Simple single plot visualizing function.

        Input:
            image: cv2 type object
            outputs: torch.tensor object returned by the predict() function

        Output:
            None
        """
        # if len(gt[f]) == 8:
        #     cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        # else:
        #     cv2.rectangle(im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 3)
        # if len(location) == 8:
        #     cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        # else:
        #     location = [int(l) for l in location]  #
        #     cv2.rectangle(im, (location[0], location[1]),
        #                     (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
        # cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # cv2.imshow(video['name'], im)
        # cv2.waitKey(1)

        #above code taken from DaSiamRPN's viz

        raise NotImplementedError("Implement me!")
     

    def init_bbox(self,image):
        """
        Visual helper function that primes the object tracker by returning a set of bounding
        box coordinates of a target to be tracked in a given image.

        Input:
            X

        Output:
            Y
        """
        return self.MS.select(image)