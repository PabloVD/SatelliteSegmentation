import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.helpers import show_mask
from satellite_segmentation.segmenters import ExcessGreenSegmenter, LangSamSegmenter, SamEGSegmenter

def tree_segmentation(image_path: str, mode: str):

    print(f"Running {mode} segmentation")

    if mode=="eg":
        segmenter = ExcessGreenSegmenter(image_path)
    elif mode=="sameg":
        segmenter = SamEGSegmenter(image_path)
    elif mode=="langsam":
        segmenter = LangSamSegmenter(image_path, text_prompt="tree")
    
    mask = segmenter.predict()
    segmenter.mask2geojson("trees_sanvicent.geojson")

    return mask

if __name__=="__main__":

    image_path = "../images/satellite_sanvicent.tif"

    mask_eg         = tree_segmentation(image_path, "eg")
    mask_sameg      = tree_segmentation(image_path, "sameg")
    mask_langsam    = tree_segmentation(image_path, "langsam")

    img = Image.open(image_path)
    plt.figure(figsize=(15, 10),layout="tight")

    plt.subplot(1, 3, 1)
    plt.title("Excess green")
    plt.imshow(img)
    show_mask(mask_eg, plt.gca(),borders=False)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Excess green + Sam2")
    plt.imshow(img)
    show_mask(mask_sameg, plt.gca(),borders=False)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("LangSam")
    plt.imshow(img)
    show_mask(mask_langsam, plt.gca(),borders=False)
    plt.axis('off')

    plt.show()
