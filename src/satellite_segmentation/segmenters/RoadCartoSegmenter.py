
import cv2
from PIL import Image
import numpy as np

from .Segmenter import Segmenter

class RoadCartoSegmenter(Segmenter):

    def __init__(self, image_path: str):
        
        super().__init__(image_path)
        image = Image.open(image_path)
        self.image = np.array(image)

    def predict(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # Roads in Carto Voyager tiles are white
        # Apply a filter of white areas
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

        mask = (mask > 0).astype("uint8")

        self.mask = mask

        return mask
