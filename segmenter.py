from samgeo import tms_to_geotiff, raster_to_vector
from samgeo.text_sam import LangSAM

class Segmenter():

    def __init__(self,
                 bbox: list[float],
                 zoom: int,
                 text_prompt: str,
                 image = "satellite.tif"):
        
        self.image = image
        
        tms_to_geotiff(output=image, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)

        self.sam = LangSAM()

    def predict(self, text_prompt):

        self.sam.predict(self.image, text_prompt, box_threshold=0.24, text_threshold=0.24)
        raster_to_vector("masks.tif", "masks.geojson")


if __name__=="__main__":

    lat_min = 41.383
    lon_min = 2.155
    lat_max = 41.387
    lon_max = 2.165
    zoom = 18
    bbox = [lon_min, lat_min, lon_max, lat_max]
    text_prompt = "tree"

    segmenter = Segmenter(bbox, zoom, text_prompt)