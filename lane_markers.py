import matplotlib.pyplot as plt
from helpers import show_mask
from PIL import Image
from LaneSegmenter import LaneSegmenter

if __name__=="__main__":

    image_path = "samgeo_tests/satellite_sanvicent.tif"
    segmenter = LaneSegmenter(image_path)
    mask = segmenter.predict()
    segmenter.mask2geojson("lanes_sanvicent.geojson")

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("Lane markers")
    plt.show()
