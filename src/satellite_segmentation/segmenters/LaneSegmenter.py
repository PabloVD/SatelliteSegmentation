
import cv2

from .Segmenter import Segmenter
from .RoadCartoSegmenter import RoadCartoSegmenter
from .RoadSamSegmenter import RoadSamSegmenter
from ..utils import download_cartovoyager, get_latlon_bounds

class LaneSegmenter(Segmenter):

    def __init__(self, image_path: str, use_mask_init: bool = True):
        
        super().__init__(image_path)

        self.image = cv2.imread(image_path)
        self.use_mask_init = use_mask_init
        self.bbox = get_latlon_bounds(image_path)
        self.color_filter = False

        self.roadsam = RoadSamSegmenter(image_path)
        
    def get_carto_road_mask(self):

        carto_image_path = "cartovoyager_roads.tif"
        download_cartovoyager(carto_image_path, bbox=self.bbox)
        road_segmenter = RoadCartoSegmenter(carto_image_path)
        mask = road_segmenter.predict()

        return mask
    
    def get_sam2_road_mask(self, mask_init):

        w, h = self.image.shape[1], self.image.shape[0]
        mask_init = cv2.resize(mask_init, (w, h))
        mask_init = self.roadsam.predict(mask_init=mask_init)

        return mask_init

    def predict(self):

        if self.use_mask_init:
            
            mask_init = self.get_carto_road_mask()
            mask_init = self.get_sam2_road_mask(mask_init)
            image = cv2.bitwise_and(self.image, self.image, mask=mask_init)

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

        # Canny edge detection
        edges = cv2.Canny(masked, threshold1=50, threshold2=150, apertureSize=3)

        # Morphological closing operations to clean up
        # Dilate to close gaps, then erode to thin
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        lane_mask = (cleaned > 0).astype("uint8")
        self.mask = lane_mask
        
        return lane_mask
