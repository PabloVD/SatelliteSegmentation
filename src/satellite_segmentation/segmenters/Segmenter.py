import rasterio
import rasterio
import geopandas as gpd
from shapely import simplify
from shapely.geometry import shape
from rasterio.features import shapes
from abc import ABC, abstractmethod
from ..helpers import num_points_geodataframe
class Segmenter(ABC):

    def __init__(self, image_path: str):

        self.image_path = image_path
        self.image, self.transform, self.crs = self.load_image()
        self.mask = None
        self.simplify_tolerance = 1.e-5

    def load_image(self):

        with rasterio.open(self.image_path) as src:
            image = src.read()
            transform = src.transform
            crs = src.crs

        return image, transform, crs
    
    def mask2geojson(self, file_name: str, simplify: bool = True):

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

        # Simplify exported polylines using the Douglas-Peucker algorithm
        if simplify:
            gdf = self.simplify_polygons(gdf)

        # Save to GeoJSON
        gdf.to_file(file_name, driver="GeoJSON")

    def simplify_polygons(self, gdf):
        new_gdf = gdf.copy()

        for i in range(len(gdf["geometry"])):
            geo = gdf["geometry"][i]
            new_geo = simplify(geo, tolerance=self.simplify_tolerance)
            new_gdf["geometry"][i] = new_geo

        return new_gdf

    @abstractmethod
    def predict(self):
        pass