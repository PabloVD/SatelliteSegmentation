# These helpers functions are adapted from the Sam2 notebooks

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform_bounds
from samgeo import tms_to_geotiff

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        # if len(scores) > 1:
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def num_points_geodataframe(gdf):
    return sum([len(geo.exterior.coords) for geo in gdf["geometry"]])

def download_cartovoyager(image_path, bbox):
    source = "https://a.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    zoom = 18
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source=source, overwrite=True)

import numpy as np

def sample_n_points_from_mask(mask: np.ndarray, N: int) -> np.ndarray:
    """
    Evenly sample N points from a binary mask where nonzero pixels are valid.

    Args:
        mask (np.ndarray): Binary mask (nonzero = valid region).
        N (int): Number of points to sample.

    Returns:
        np.ndarray: Array of shape (N, 2), each row is (row, col) of a sampled point.
    """
    # Get all valid (nonzero) coordinates
    coords = np.column_stack(np.where(mask > 0))
    total = len(coords)
    
    if total == 0:
        raise ValueError("No valid pixels found in the mask.")
    if N >= total:
        # Return all if mask has fewer points than N
        return coords

    # Even sampling by selecting every (total / N)-th point
    indices = np.linspace(0, total - 1, N, dtype=int)
    print(len(indices))
    return coords[indices]

def get_latlon_bounds(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # in native CRS
        crs = src.crs

        # Transform to WGS84 (EPSG:4326) if needed
        if crs.to_string() != 'EPSG:4326':
            bounds_wgs84 = transform_bounds(crs, 'EPSG:4326', *bounds)
        else:
            bounds_wgs84 = bounds

        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        bbox = [min_lon, min_lat, max_lon, max_lat]
        return bbox
