import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.utils import show_mask
from satellite_segmentation.segmenters import SamEGSegmenter

if __name__=="__main__":

    image_path = "../images/uab3.tif"
    segmenter = SamEGSegmenter(image_path,
                               sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt",
                               sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml")
    mask = segmenter.predict()
    segmenter.mask2geojson("sameg_trees_uab.geojson")

    img = Image.open(image_path)
    plt.figure(figsize=(10, 10),layout="tight")
    plt.imshow(img)
    show_mask(mask, plt.gca(),borders=False)
    plt.axis('off')
    plt.title("Excess green + Sam2")
    plt.show()
