
import cv2

from .Segmenter import Segmenter
from .LangSamSegmenter import LangSamSegmenter

class LaneSegmenter(Segmenter):

    def __init__(self, image_path: str, use_mask_init: bool = True):
        
        super().__init__(image_path)

        self.image = cv2.imread(image_path)
        self.use_mask_init = use_mask_init
        self.langsam = LangSamSegmenter(image_path, text_prompt="road")

    def predict(self):

        if self.use_mask_init:
            mask = self.langsam.predict()
            image = cv2.bitwise_and(self.image, self.image, mask=mask)
        else:
            image = self.image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # White mask: high V, low S
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
        # Optional: Yellow mask (if lines are yellow in some countries)
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))

        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        masked = cv2.bitwise_and(gray, gray, mask=lane_mask)

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
