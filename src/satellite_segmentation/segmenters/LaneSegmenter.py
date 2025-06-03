
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .Segmenter import Segmenter
from .LangSamSegmenter import LangSamSegmenter
from .RoadCartoSegmenter import RoadCartoSegmenter
from .RoadSamSegmenter import RoadSamSegmenter
from ..helpers import download_cartovoyager

class LaneSegmenter(Segmenter):

    def __init__(self, image_path: str, use_mask_init: bool = True, bbox : list[float] = None):
        
        super().__init__(image_path)

        self.image = cv2.imread(image_path)
        self.use_mask_init = use_mask_init
        if bbox is None:
            self.langsam = LangSamSegmenter(image_path, text_prompt="road")
        self.bbox = bbox
        self.color_filter = False

        self.roadsam = RoadSamSegmenter(image_path)
        
    def roads_cartovoyager(self):

        image_path = "cartovoyager_roads.tif"
        download_cartovoyager(image_path, bbox=self.bbox)
        road_segmenter = RoadCartoSegmenter(image_path)
        mask = road_segmenter.predict()

        return mask

    def predict(self):

        if self.use_mask_init:
            if self.bbox is None:
                mask_init = self.langsam.predict()
            else:
                mask_init = self.roads_cartovoyager()
                w, h = self.image.shape[1], self.image.shape[0]
                mask_init = cv2.resize(mask_init, (w, h))
                mask_init = self.roadsam.predict(mask_init=mask_init)
            image = cv2.bitwise_and(self.image, self.image, mask=mask_init)
            
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 3, 1)
            # # plt.title("Input")
            # plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            # plt.subplot(1, 3, 2)
            # # plt.title("Edges")
            # plt.imshow(mask_init, cmap="gray")
            # plt.subplot(1, 3, 3)
            # # plt.title("Final Lane Mask")
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.tight_layout()
            # plt.show()
            # # exit()

        else:
            image = self.image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.color_filter:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # White mask: high V, low S
            white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
            # Optional: Yellow mask (if lines are yellow in some countries)
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))

            lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

            masked = cv2.bitwise_and(gray, gray, mask=lane_mask)

        else:
            masked = gray

        # w, h = self.image.shape[1], self.image.shape[0]
        # masked = cv2.resize(masked, dsize=None, fx=1.2, fy=1.2)

        # Canny edge detection
        edges = cv2.Canny(masked, threshold1=50, threshold2=150, apertureSize=3)

        # Morphological operations to clean up
        # Dilate to close gaps, then erode to thin (closing operation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        cleaned = cv2.erode(dilated, kernel, iterations=1)

        lane_mask = (cleaned > 0).astype("uint8")
        self.mask = lane_mask
        
        return lane_mask
