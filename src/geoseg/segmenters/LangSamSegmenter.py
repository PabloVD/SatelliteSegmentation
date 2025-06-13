from samgeo.text_sam import LangSAM

from .Segmenter import Segmenter

class LangSamSegmenter(Segmenter):

    def __init__(self,
                 image_path: str,
                 text_prompt: str,
                 threshold: int = 0.2,
                 masktif_path: str = "masks.tif",
                 ):
        
        super().__init__(image_path)

        self.text_prompt = text_prompt
        self.masktif_path = masktif_path
        self.threshold = threshold

        self.langsam = LangSAM()

    def predict(self):

        mask, boxes, phrases, logits = self.langsam.predict(self.image_path,
                                       self.text_prompt,
                                       output = self.masktif_path,
                                       box_threshold = self.threshold,
                                       text_threshold = self.threshold,
                                       return_results = True
                                       )
        
        mask = mask.sum(axis=0)
        mask = mask.squeeze()
        mask = mask.numpy()

        mask = (mask > 0).astype("uint8")

        self.mask = mask

        return mask