from detectree import Classifier

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

        self.mask = mask

        return mask
