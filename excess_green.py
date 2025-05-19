import rasterio
import numpy as np

image_path = "samgeo_tests/satellite.tif"
#image_path = "satellite.tif"

# Load the image
with rasterio.open(image_path) as src:
    image = src.read()  # shape: (bands, height, width)
    transform = src.transform
    crs = src.crs

r = image[0].astype(float)
g = image[1].astype(float)
b = image[2].astype(float)
exg = 2 * g - r - b
vegetation_mask = exg > 20  # Adjust threshold empirically

from rasterio import features

with rasterio.open("vegetation_mask.tif", "w", 
                   driver="GTiff", height=vegetation_mask.shape[0],
                   width=vegetation_mask.shape[1], count=1, dtype="uint8",
                   crs=crs, transform=transform) as dst:
    dst.write(vegetation_mask.astype("uint8"), 1)

import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

# Extract shapes from mask
results = (
    {"geometry": shape(geom), "value": value}
    for geom, value in shapes(
        vegetation_mask.astype("uint8"),
        mask=vegetation_mask,
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
gdf.to_file("vegetation.geojson", driver="GeoJSON")

