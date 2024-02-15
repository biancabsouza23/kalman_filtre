import numpy as np
import cv2 as cv

class Kalman():
    """
    Based on:
    https://www.bacancytechnology.com/qanda/python/opencv-kalman-filter-with-python
    https://pieriantraining.com/kalman-filter-opencv-python-example/
    """
    def __init__(self, size: int, initial_positions: np.ndarray = None):
        self.size = size
        if initial_positions == None:
            initial_positions = np.zeros((size, 2), np.float32)
        self.kalmans = [cv.KalmanFilter(2, 2) for _ in range(size)]
        for k, (x, y) in zip(self.kalmans, initial_positions):
            k.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
            k.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
            k.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
            k.statePre = np.array([x, y], np.float32)
            k.statePost = np.array([x, y], np.float32)