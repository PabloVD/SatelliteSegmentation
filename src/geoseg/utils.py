# The mask plotting helpers are adapted from the Sam2 notebooks

import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
import cv2
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

def show_masks(image, masks, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        plt.axis('off')
        plt.show()

def num_points_geodataframe(gdf):
    return sum([len(geo.exterior.coords) for geo in gdf["geometry"]])

def download_carto(image_path, bbox):
    source = "https://a.basemaps.cartocdn.com/rastertiles/light_nolabels/{z}/{x}/{y}.png"
    zoom = 18
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source=source, overwrite=True)

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

def split_grayscale_array_to_tiles(np_array, tile_size=1024, output_dir="tiles"):
    os.makedirs(output_dir, exist_ok=True)

    if np_array.ndim != 2:
        raise ValueError("Input array must be 2D (grayscale).")

    height, width = np_array.shape
    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)

    for row in range(rows):
        for col in range(cols):
            top = row * tile_size
            left = col * tile_size
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)

            tile_data = np_array[top:bottom, left:right]

            # Pad if necessary
            if tile_data.shape != (tile_size, tile_size):
                padded_tile = np.zeros((tile_size, tile_size), dtype=np_array.dtype)
                padded_tile[:tile_data.shape[0], :tile_data.shape[1]] = tile_data
                tile_data = padded_tile

            tile_image = Image.fromarray(tile_data, mode='L')
            tile_image.save(
                os.path.join(output_dir, f"tile_{row}_{col}.tiff"),
                compression="none"
            )

def extract_connected_extension(initial_mask: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    """
    Keeps only those regions in refined_mask that are connected to initial_mask.
    
    Parameters:
        initial_mask: binary np.ndarray (H x W), trusted topology (0/1 or 0/255)
        refined_mask: binary np.ndarray (H x W), possibly noisy (0/1 or 0/255)
        
    Returns:
        filtered_mask: binary np.ndarray (H x W), refined + topologically consistent
    """
    # Ensure 0/255 for OpenCV floodFill
    refined_bin = (refined_mask > 0).astype(np.uint8) * 255
    initial_bin = (initial_mask > 0).astype(np.uint8) * 255

    # Prepare mask for floodFill (needs padding)
    h, w = refined_bin.shape
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    # Output placeholder
    connected = np.zeros_like(refined_bin)

    # Run flood fill from every pixel in the initial mask
    seeds = np.column_stack(np.where(initial_bin == 255))
    for y, x in seeds:
        if refined_bin[y, x] == 255 and connected[y, x] == 0:
            temp = np.zeros_like(refined_bin)
            temp_mask = mask_ff.copy()
            cv2.floodFill(refined_bin, temp_mask, (x, y), 128)
            filled = (refined_bin == 128).astype(np.uint8) * 255
            connected = cv2.bitwise_or(connected, filled)
            refined_bin[refined_bin == 128] = 255  # reset for next seed

    return (connected > 0).astype(np.uint8)