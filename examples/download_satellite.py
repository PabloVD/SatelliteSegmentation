from samgeo import tms_to_geotiff

image_path = "../images/uab_rondabout_18.tif"

# Bound box from this API call
# https://overpass-api.de/api/map?bbox=2.108147,41.495556,2.109847,41.496922
bbox = [2.108147,41.495556,2.109847,41.496922]
zoom = 18
source = "Satellite"

tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source=source, overwrite=True)
