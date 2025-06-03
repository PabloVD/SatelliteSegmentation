from detectree import Classifier
import cv2

from .Segmenter import Segmenter

class DetecTreeSegmenter(Segmenter):

    def __init__(self,
                 image_path: str,
                 ):
        
        super().__init__(image_path)

        self.detectree = Classifier()

    def predict(self):

        mask = self.detectree.predict_img(self.image_path)

        mask = (mask > 0).astype("uint8")
        mask *= 255

        # Apply closing to avoid sparse structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = (mask > 0).astype("uint8")

        self.mask = mask

        return mask
