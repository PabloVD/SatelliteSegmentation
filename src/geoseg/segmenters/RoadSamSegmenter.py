
import cv2
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.morphology import skeletonize
from geoseg.utils import show_masks

from .Segmenter import Segmenter
from .RoadCartoSegmenter import RoadCartoSegmenter
from ..utils import sample_n_points_from_mask

class RoadSamSegmenter(Segmenter):

    def __init__(self, image_path: str):
        
        super().__init__(image_path)

        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        device = "cuda"

        self.sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        image = Image.open(image_path)
        self.image = np.array(image.convert("RGB"))
        self.predictor.set_image(self.image)

    def predict(self, mask_init: np.ndarray | None = None, debug: bool = False) -> np.ndarray:

        # skeleton = skeletonize(mask_init > 0)

        # # Get skeleton coordinates
        # skel_coords = np.column_stack(np.where(skeleton))
        # print(skel_coords.shape)

        # # Sample every N-th skeleton pixel
        # stride = 100
        # points_coords = skel_coords[::stride]

        if mask_init is None:
            road_carto = RoadCartoSegmenter(self.image_path)
            mask_init = road_carto.predict()

        points_coords = sample_n_points_from_mask(mask_init, N=20)
        points_coords = np.array([[point[1],point[0]] for point in points_coords])
        point_labels = np.ones(len(points_coords))

        low_res_mask = cv2.resize(mask_init, (256, 256))

        masks, scores, logits = self.predictor.predict(
            point_coords=points_coords,
            point_labels=point_labels,
            # mask_input=low_res_mask[None, :, :],
            box=None,
            multimask_output=False,
        )

        if debug:
            show_masks(self.image, masks, point_coords=points_coords, input_labels=point_labels)

        mask = masks.sum(axis=0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        
        # Closing (dilation + erosion)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Opening (erosion + dilation)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

        mask = (mask > 0).astype("uint8")

        self.mask = mask

        return mask
