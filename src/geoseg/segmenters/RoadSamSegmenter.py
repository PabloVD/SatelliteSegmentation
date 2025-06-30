
import cv2
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.morphology import skeletonize
from geoseg.utils import show_masks

from .Segmenter import Segmenter
from .RoadCartoSegmenter import RoadCartoSegmenter
from ..utils import sample_n_points_from_mask, extract_connected_extension

from skimage.transform import probabilistic_hough_line
from skimage.feature import canny



    
def mock_soft_logits(binary_mask, sharpness=10.0, blur_kernel=5):
    # Convert to float
    float_mask = binary_mask.astype(np.float32)
    
    # Blur the mask (simulates uncertainty at edges)
    blurred = cv2.GaussianBlur(float_mask, (blur_kernel, blur_kernel), 0)
    
    # Stretch blurred mask from [0,1] → [-sharpness/2, +sharpness/2]
    logits = sharpness * (blurred - 0.5)
    return logits

class RoadSamSegmenter(Segmenter):

    def __init__(self, image_path: str, refine=False):
        
        super().__init__(image_path)

        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        device = "cuda"

        self.sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        image = Image.open(image_path)
        self.image = np.array(image.convert("RGB"))
        self.predictor.set_image(self.image)

        self.refine = refine

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

        n_sample_points = 30

        points_coords = sample_n_points_from_mask(mask_init, N=n_sample_points)
        points_coords = np.array([[point[1],point[0]] for point in points_coords])
        point_labels = np.ones(len(points_coords))

        # low_res_mask = cv2.resize(mask_init, (256, 256))
        # logits = mock_soft_logits(low_res_mask)

        masks, scores, logits = self.predictor.predict(
            point_coords=points_coords,
            point_labels=point_labels,
            # mask_input=logits[None, :, :],
            box=None,
            multimask_output=False,
        )

        # debug = True

        if debug:
            # show_masks(self.image, mask_init[None, :, :], point_coords=points_coords, input_labels=point_labels)
            show_masks(self.image, masks, point_coords=points_coords, input_labels=point_labels)

        mask = masks.sum(axis=0)

        mask = extract_connected_extension(mask_init, mask)

        mask = cv2.bitwise_or(mask, mask_init)

        kernel_rad = 10

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_rad, kernel_rad))
        
        # Closing (dilation + erosion)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Opening (erosion + dilation)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

        # # Apply Gaussian blur
        # blurred = cv2.GaussianBlur(mask*255, (7, 7), sigmaX=0)

        # # Re-threshold to get back to binary mask if you want hard edges
        # _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        mask = (mask > 0).astype("uint8")

        if self.refine:

            mask = self.refinement(mask, mask_init)

        self.mask = mask

        return mask
    
    def refinement(self, mask_sam2, mask_initial):

        edges = canny(mask_sam2.astype(np.uint8)*255, sigma=2)
        edges = edges.astype(np.uint8) * 255
        lines = probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=5)

        hough_mask = np.zeros_like(mask_sam2)
        for p0, p1 in lines:
            cv2.line(hough_mask, p0, p1, color=255, thickness=1)

        # cv2.imshow("Edges",edges)
        # cv2.waitKey(0) 
        # cv2.imshow("Hough",hough_mask)
        # cv2.waitKey(0) 
        # # show_masks(self.image, hough_mask)
        # # show_masks(self.image, edges)
        # exit()

        blurred = cv2.medianBlur(mask_sam2.astype(np.uint8)*255, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                param1=100, param2=30, minRadius=10, maxRadius=2000)
        # print(circles)

        circle_mask = np.zeros_like(mask_sam2)
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for x, y, r in circles:
                cv2.circle(circle_mask, (x, y), r, 255, thickness=1)

        # cv2.imshow("Hough",hough_mask)
        # cv2.waitKey(0) 
        # cv2.imshow("Circle",circle_mask*255)
        # cv2.waitKey(0)
        # exit()

        refined_mask = (hough_mask | circle_mask | mask_initial).astype(np.uint8)

        return refined_mask
