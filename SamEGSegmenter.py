from scipy import ndimage
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from Segmenter import Segmenter
from ExcessGreenSegmenter import ExcessGreenSegmenter

class SamEGSegmenter(Segmenter):

    def __init__(self,
                 image_path: str,
                 threshold: int = 20,
                 morph_kernel_size: int = 5,
                 device = "cuda",
                 sam2_checkpoint = "/home/tda/CARLA/DigitalTwins/segment_tool/samgeo_tests/sam2.1_hiera_small.pt",
                 sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
                 ):
        
        super().__init__(image_path)

        self.ExGreen = ExcessGreenSegmenter(image_path, threshold, morph_kernel_size)

        self.sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        image = Image.open(image_path)
        self.image = np.array(image.convert("RGB"))
        self.predictor.set_image(self.image)

    def get_bounding_boxes(self, binary_mask):
        """
        Takes a binary mask and returns a list of bounding boxes.
        Each bounding box is in the format (min_row, min_col, max_row, max_col).
        """
        labeled_mask, num_features = ndimage.label(binary_mask)
        objects = ndimage.find_objects(labeled_mask)
        
        bounding_boxes = []
        for obj_slice in objects:
            if obj_slice is not None:
                min_row, max_row = obj_slice[1].start, obj_slice[1].stop
                min_col, max_col = obj_slice[0].start, obj_slice[0].stop
                bounding_boxes.append((min_row, min_col, max_row, max_col))
        
        bounding_boxes = np.array(bounding_boxes)
        
        return bounding_boxes

    def predict(self):

        self.mask_init = self.ExGreen.predict()

        bboxes = self.get_bounding_boxes(self.mask_init)

        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bboxes,
            multimask_output=False,
        )

        masks = masks.sum(axis=0)
        mask = masks[0]

        mask = (mask > 0).astype("uint8")

        self.mask = mask

        return mask