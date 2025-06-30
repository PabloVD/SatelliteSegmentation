
import cv2
from PIL import Image
import numpy as np

from .Segmenter import Segmenter
from ..utils import download_carto, get_latlon_bounds

class RoadCartoSegmenter(Segmenter):

    def __init__(self, image_path: str):
        
        super().__init__(image_path)

        carto_image_path = "cartovoyager_roads.tif"

        bbox = get_latlon_bounds(image_path)

        download_carto(carto_image_path, bbox=bbox)

        carto_image = Image.open(carto_image_path)
        self.carto_image = np.array(carto_image)

    def predict(self):

        # Roads in Carto Voyager tiles are white and yellow
        # Apply a filter of these areas
        gray = cv2.cvtColor(self.carto_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

        # Resize to the original satellite image, since the Carto Voyager tile is low res
        w, h = self.image.shape[2], self.image.shape[1]
        mask = cv2.resize(mask, (w, h))

        mask = (mask > 0).astype("uint8")

        self.mask = mask

        return mask
