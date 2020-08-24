"""
Base class script for the 1stDayKit functions, methods and modules.
"""


class BaseClass(object):
    """
    Inherit from me and follow my structure!
    """

    def __init__(self,name,device='cpu'):
        
        self.name = name
        self.device = device
        

    def predict(self):
        raise NotImplementedError("To be implemented in individual module's script")


    def prime(self):
        raise NotImplementedError("To be implemented in individual module's script")
    

    def visualize(self):
        raise NotImplementedError("To be implemented in individual module's script")




