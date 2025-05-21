import rasterio
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes


def excess_green_segmentation(image_path: str, threshold: int = 20):

    # Load the image
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs

    r = image[0].astype(float)
    g = image[1].astype(float)
    b = image[2].astype(float)
    exg = 2 * g - r - b
    vegetation_mask = exg > threshold  # Adjust threshold empirically

    mask2geojson(vegetation_mask, transform, crs)

def mask2geojson(mask, transform, crs):

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
    gdf.to_file("vegetation.geojson", driver="GeoJSON")

if __name__=="__main__":

    image_path = "samgeo_tests/satellite.tif"
    #image_path = "satellite.tif"

    excess_green_segmentation(image_path)

