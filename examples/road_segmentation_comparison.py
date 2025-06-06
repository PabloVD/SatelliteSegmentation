import matplotlib.pyplot as plt
from PIL import Image

from satellite_segmentation.utils import show_mask
from satellite_segmentation.segmenters import Segmenter, RoadCartoSegmenter, RoadSamSegmenter, LangSamSegmenter

if __name__=="__main__":

    image_path = "../images/uab_rotonda22.tif"

    modes = {
        "Carto Voyager":    RoadCartoSegmenter(image_path),
        "Carto + Sam2":     RoadSamSegmenter(image_path),
        "LangSam":          LangSamSegmenter(image_path, text_prompt="road"),
    }

    n_modes = len(modes.keys())

    img = Image.open(image_path)
    plt.figure(figsize=(5*n_modes, 5),layout="tight")

    for i, mode in enumerate(modes.keys()):

        print(f"Running {mode} segmentation")

        segmenter: Segmenter = modes[mode]
        
        mask = segmenter.predict()
    
        plt.subplot(1, n_modes, i+1)
        plt.title(mode)
        plt.imshow(img)
        show_mask(mask, plt.gca(),borders=False)
        plt.axis('off')

    plt.savefig("../images/road_segmentation_comparison.png")
    plt.show()
