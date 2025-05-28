import matplotlib.pyplot as plt
from helpers import show_mask
from PIL import Image
from LangSamSegmenter import LangSamSegmenter

if __name__=="__main__":

    image_path = "samgeo_tests/satellite_sanvicent.tif"
    segmenter = LangSamSegmenter(image_path, text_prompt="tree")
    mask = segmenter.predict()
    segmenter.mask2geojson("langsam_trees_sanvicent.geojson")

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("LangSam")
    plt.show()
