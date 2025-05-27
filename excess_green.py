import rasterio
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
import matplotlib.pyplot as plt
import numpy as np
from helpers import show_mask
from PIL import Image

def load_image(image_path: str):

    # Load the image
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs

    return image, transform, crs

def excess_green_segmentation(image, threshold: int = 20):

    r = image[0].astype(float)
    g = image[1].astype(float)
    b = image[2].astype(float)
    exg = 2 * g - r - b
    vegetation_mask = exg > threshold  # Adjust threshold empirically

    return vegetation_mask

def mask2geojson(mask, transform, crs, file_name="vegetation.geojson"):

    # Extract shapes from mask
    results = (
        {"geometry": shape(geom), "value": value}
        for geom, value in shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=transform
        )
        if value == 1
    )

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame.from_records(results)

    # Set geometry column and CRS
    gdf = gdf.set_geometry("geometry")
    gdf.set_crs(crs, inplace=True)

    # Save to GeoJSON
    gdf.to_file(file_name, driver="GeoJSON")

if __name__=="__main__":

    image_path = "samgeo_tests/satellite.tif"
    #image_path = "satellite.tif"

    image, transform, crs = load_image(image_path)

    mask = excess_green_segmentation(image)

    mask2geojson(mask, transform, crs)

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("Excess green")
    plt.show()

