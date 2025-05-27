import sys
from samgeo import tms_to_geotiff, raster_to_vector
from samgeo.text_sam import LangSAM
import argparse
import pathlib

def run_langsam(bbox, zoom, threshold=0.24):

    outpath = pathlib.Path("outputs")
    outpath.mkdir(parents=True, exist_ok=True)

    text_prompt = "tree"

    image_path = str(outpath / "satellite.tif")
    masktif_path = str(outpath / "masks.tif")
    maskgeojson_path = str(outpath / "masks.geojson")

    tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)

    sam = LangSAM()

    sam.predict(image_path, text_prompt, output=masktif_path, box_threshold=threshold, text_threshold=threshold)
    raster_to_vector(masktif_path, maskgeojson_path)

def main():

    print(sys.version)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lon_min')
    parser.add_argument('--lat_min')
    parser.add_argument('--lon_max')
    parser.add_argument('--lat_max')
    args = parser.parse_args()

    lon_min, lat_min, lon_max, lat_max = args.lon_min, args.lat_min, args.lon_max, args.lat_max
    
    print("Args:", args)
    print("Coords:", lon_min, lat_min, lon_max, lat_max)
    print("Received sys.argv:", sys.argv)

    lat_min = 41.383
    lon_min = 2.155
    lat_max = 41.387
    lon_max = 2.165

    zoom = 18
    threshold = 0.24
    bbox = [lon_min, lat_min, lon_max, lat_max]
    bbox = [-0.376748,39.455013,-0.374227,39.457655]
    
    run_langsam(bbox, zoom, threshold)


if __name__ == '__main__':
    main()

    