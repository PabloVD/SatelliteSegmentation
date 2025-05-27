from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from helpers import show_masks, show_mask, show_box
from excess_green import excess_green_segmentation, load_image, mask2geojson
from shapely.geometry import Point, Polygon
import geopandas as gpd
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from scipy import ndimage

def get_bounding_boxes(binary_mask):
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

def excess_green_pipeline(image_path):

    image, transform, crs = load_image(image_path)

    mask_init = excess_green_segmentation(image)

    bboxes = get_bounding_boxes(mask_init)

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask_init, plt.gca(),borders=False)
    # for box in bboxes:
    #     show_box(box, plt.gca())
    plt.axis('off')
    plt.title("Excess green")
    plt.show()
    
    return bboxes

device = "cuda"

image_path = 'samgeo_tests/satellite_sanvicent.tif'

image = Image.open(image_path)
# image.show()
image = np.array(image.convert("RGB"))

# cv2.imshow("img",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

sam2_checkpoint = "/home/tda/CARLA/DigitalTwins/segment_tool/samgeo_tests/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

bboxes = point_coords = excess_green_pipeline(image_path)

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=bboxes,
    multimask_output=False,
)

masks = masks.sum(axis=0)

sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

point_coords = None
point_labels = None

img = Image.open(image_path)
plt.figure(figsize=(10, 10),layout="tight")
plt.imshow(img)
show_mask(masks[0], plt.gca(),borders=False)
# for box in bboxes:
#     show_box(box, plt.gca())
plt.axis('off')
plt.title("Excess green + Sam2")
plt.show()