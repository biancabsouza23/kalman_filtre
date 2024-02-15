import cv2 as cv
import numpy as np

class Webcam:
    """
    Classe para captura de imagens da webcam.
    """
    def __init__(self):
        self.webcam = None
        self.__start__()
        
    def __start__(self):
        self.webcam = cv.VideoCapture(0)
        
    def __stop__(self):
        self.webcam.release()
    
    def __del__(self):
        self.__stop__()
    
    def take_image(self) -> np.ndarray:
        """
        Captura uma imagem da webcam.
        """
        _, image = self.webcam.read()
        return image