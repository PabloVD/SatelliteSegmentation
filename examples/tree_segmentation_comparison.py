import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.helpers import show_mask
from satellite_segmentation.segmenters import Segmenter, ExcessGreenSegmenter, LangSamSegmenter, SamEGSegmenter, DetecTreeSegmenter


if __name__=="__main__":

    image_path = "../images/satellite_sanvicent.tif"
    image_path = "../uab_rotonda.tif"

    modes = {
        "Excess Green":         ExcessGreenSegmenter(image_path),
        "Excess Green + Sam2":  SamEGSegmenter(image_path, sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"),
        "LangSam":              LangSamSegmenter(image_path, text_prompt="tree"),
        "Detectree":            DetecTreeSegmenter(image_path)
    }

    n_modes = len(modes.keys())

    img = Image.open(image_path)
    plt.figure(figsize=(5*n_modes, 10),layout="tight")

    for i, mode in enumerate(modes.keys()):

        print(f"Running {mode} segmentation")

        segmenter: Segmenter = modes[mode]
        
        mask = segmenter.predict()
    
        plt.subplot(1, n_modes, i+1)
        plt.title(mode)
        plt.imshow(img)
        show_mask(mask, plt.gca(),borders=False)
        plt.axis('off')

    plt.show()
