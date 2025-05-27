from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from helpers import show_masks, show_mask
from excess_green import excess_green_segmentation, load_image, mask2geojson
from shapely.geometry import Point, Polygon
import geopandas as gpd
from rasterio.transform import rowcol
import matplotlib.pyplot as plt

def excess_green_pipeline(image_path):

    image, transform, crs = load_image(image_path)

    mask_init = excess_green_segmentation(image)

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask_init, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("Excess green")
    plt.show()

    mask2geojson(mask_init, transform, crs)

    gdf = gpd.read_file("vegetation.geojson")

    centroids = [ [poly.centroid.x, poly.centroid.y] for poly in gdf.geometry ]

    pixel_points = [rowcol(transform, x, y) for x, y in centroids]

    point_coords = np.array(pixel_points)

    return point_coords

device = "cuda"

image_path = 'samgeo_tests/satellite.tif'

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

point_coords = excess_green_pipeline(image_path)

# Convert to SAM input format
point_labels = np.ones(len(point_coords), dtype=int)

masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

show_masks(image, masks, scores, point_coords=point_coords, input_labels=point_labels, borders=True)

# try with predict_batch