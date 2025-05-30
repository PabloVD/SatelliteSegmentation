import rasterio
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from abc import ABC, abstractmethod

class Segmenter(ABC):

    def __init__(self, image_path: str):

        self.image_path = image_path
        self.image, self.transform, self.crs = self.load_image()
        self.mask = None

    def load_image(self):

        with rasterio.open(self.image_path) as src:
            image = src.read()
            transform = src.transform
            crs = src.crs

        return image, transform, crs
    
    def mask2geojson(self, file_name):

        # Extract shapes from mask
        results = (
            {"geometry": shape(geom), "value": value}
            for geom, value in shapes(
                self.mask.astype("uint8"),
                mask=self.mask,
                transform=self.transform
            )
            if value == 1
        )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame.from_records(results)

        # Set geometry column and CRS
        gdf = gdf.set_geometry("geometry")
        gdf.set_crs(self.crs, inplace=True)
        gdf = gdf.to_crs(epsg=4326)

        # Save to GeoJSON
        gdf.to_file(file_name, driver="GeoJSON")

    @abstractmethod
    def predict(self):
        pass