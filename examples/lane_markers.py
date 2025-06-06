import matplotlib.pyplot as plt
from PIL import Image
import cv2

from satellite_segmentation.segmenters import LaneSegmenter

if __name__=="__main__":

    image_path = "../images/uab_rotonda22.tif"
    segmenter = LaneSegmenter(image_path)
    mask = segmenter.predict()
    # segmenter.mask2geojson("lanes_uab.geojson", simplify=False)

    cv2.imwrite("../images/lanes_uab_22.png", mask*255)

    img = Image.open(image_path)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Satellite")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Lane Markers Mask")
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("../images/lane_markers_segmentation.png")
    plt.show()
