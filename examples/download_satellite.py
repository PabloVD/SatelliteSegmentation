from samgeo import tms_to_geotiff

# Bound box from this API call
# https://overpass-api.de/api/map?bbox=2.108147,41.495556,2.109847,41.496922
# https://overpass-api.de/api/map?bbox=-0.369887,39.445307,-0.367892,39.446960
# https://overpass-api.de/api/map?bbox=2.108629,41.492012,2.110555,41.493495
# https://overpass-api.de/api/map?bbox=2.108383,41.495838,2.109528,41.496681
# https://overpass-api.de/api/map?bbox=2.100856,41.478160,2.105287,41.481890
bbox = [2.100856,41.478160,2.105287,41.481890]
zoom = 20
source = "Satellite"

# image_path = f"../images/satellite_rondabout_uab2_{zoom}.tif"
image_path = f"../images/cugat.tif"

tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source=source, overwrite=True)
