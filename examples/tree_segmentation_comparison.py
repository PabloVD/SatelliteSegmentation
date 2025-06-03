import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.helpers import show_mask
from satellite_segmentation.segmenters import ExcessGreenSegmenter, LangSamSegmenter, SamEGSegmenter, DetecTreeSegmenter

def tree_segmentation(image_path: str, mode: str):

    print(f"Running {mode} segmentation")

    if mode=="eg":
        segmenter = ExcessGreenSegmenter(image_path)
    elif mode=="sameg":
        segmenter = SamEGSegmenter(image_path, sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt")
    elif mode=="langsam":
        segmenter = LangSamSegmenter(image_path, text_prompt="tree")
    elif mode=="detectree":
        segmenter = DetecTreeSegmenter(image_path)
    
    mask = segmenter.predict()
    segmenter.mask2geojson("trees_sanvicent.geojson")

    return mask

if __name__=="__main__":

    image_path = "../images/satellite_sanvicent.tif"
    image_path = "../uab_rotonda.tif"

    mask_eg         = tree_segmentation(image_path, "eg")
    mask_sameg      = tree_segmentation(image_path, "sameg")
    mask_langsam    = tree_segmentation(image_path, "langsam")
    mask_detectree  = tree_segmentation(image_path, "detectree")

    img = Image.open(image_path)
    plt.figure(figsize=(20, 10),layout="tight")

    plt.subplot(1, 4, 1)
    plt.title("Excess green")
    plt.imshow(img)
    show_mask(mask_eg, plt.gca(),borders=False)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Excess green + Sam2")
    plt.imshow(img)
    show_mask(mask_sameg, plt.gca(),borders=False)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("LangSam")
    plt.imshow(img)
    show_mask(mask_langsam, plt.gca(),borders=False)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Detectree")
    plt.imshow(img)
    show_mask(mask_detectree, plt.gca(),borders=False)
    plt.axis('off')

    plt.show()
