import matplotlib.pyplot as plt
from PIL import Image

from geoseg.utils import show_mask
from geoseg.segmenters import DetecTreeSegmenter

if __name__=="__main__":

    image_path = "../images/satellite_sanvicent.tif"
    segmenter = DetecTreeSegmenter(image_path)
    mask = segmenter.predict()
    segmenter.mask2geojson("detecttree_trees_sanvicent.geojson")

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("DetectTree")
    plt.show()
