import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.utils import show_mask
from satellite_segmentation.segmenters import LangSamSegmenter

if __name__=="__main__":

    image_path = "../images/uab2.tif"
    segmenter = LangSamSegmenter(image_path, text_prompt="tree")
    mask = segmenter.predict()
    segmenter.mask2geojson("trees_uab.geojson")

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("LangSam")
    plt.show()
