import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.helpers import show_mask
from satellite_segmentation.segmenters import ExcessGreenSegmenter

if __name__=="__main__":

    image_path = "../samgeo_tests/satellite_sanvicent.tif"
    segmenter = ExcessGreenSegmenter(image_path)
    mask = segmenter.predict()
    segmenter.mask2geojson("eg_trees_sanvicent.geojson")

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("Excess green")
    plt.show()
