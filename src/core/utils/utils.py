# Write Python code here 
# import the necessary packages 
import cv2
import numpy as np

from PIL import Image


"""-------------------------------------------------------------------------------
Helper Classes
-------------------------------------------------------------------------------"""

class MouseSelector():
    """
    Helper class that takes an image and spits back out a set of bounding box coordinates
    as drawn by the user with a mouse.
    """
    def __init__(self):

        self.ref_point = [] 
        self.crop = False


    def _shape_selection(self, event, x, y, flags, param):     
        # if the left mouse button was clicked, record the starting 
        # (x, y) coordinates and indicate that cropping is being performed 
        if event == cv2.EVENT_LBUTTONDOWN: 
            self.ref_point = [(x, y)] 
    
        # check to see if the left mouse button was released 
        elif event == cv2.EVENT_LBUTTONUP: 
            # record the ending (x, y) coordinates and indicate that 
            # the cropping operation is finished 
            self.ref_point.append((x, y)) 
    
            # draw a rectangle around the region of interest 
            cv2.rectangle(self.image, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2) 
            cv2.imshow("base image", self.image) 
  

    def select(self,image):

        #Set image and create window
        self.image = image
        clone = self.image.copy() 
        cv2.namedWindow("image") 
        cv2.setMouseCallback("image", self._shape_selection) 
    
    
        # keep looping until the 'q' key is pressed 
        while True: 
            # display the image and wait for a keypress 
            cv2.imshow("image", self.image) 
            key = cv2.waitKey(1) & 0xFF
        
            # press 'r' to reset the window 
            if key == ord("r"): 
                self.image = clone.copy() 
        
            # if the 'c' key is pressed, break from the loop 
            elif key == ord("c"): 
                break
        
        if len(self.ref_point) == 2: 
            crop_img = clone[self.ref_point[0][1]:self.ref_point[1][1], self.ref_point[0][0]: 
                                                                self.ref_point[1][0]] 
            cv2.imshow("crop_img", crop_img) 
            cv2.waitKey(0) 
        
        # close all open windows 
        cv2.destroyAllWindows() 


        return [self.ref_point[0][1],
                self.ref_point[1][1],
                self.ref_point[0][0],
                self.ref_point[1][0]]


"""-------------------------------------------------------------------------------
Helper Functions
-------------------------------------------------------------------------------"""

def pil_to_cv2(img):
    """
    Converts PIL image to cv2 image
    """

    img_ = np.array(img)
    img_ = img_[:,:,::-1]

    return img_


def cv2_to_pil(img):
    """
    Converts cv2 image to PIL image
    """

    img_ = img[:,:,::-1]
    img_ = Image.fromarray(img_)

    return img_
