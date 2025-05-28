from scipy import ndimage
import numpy as np

from satellite_segmentation.Segmenter import Segmenter

class ExcessGreenSegmenter(Segmenter):

    def __init__(self, image_path, threshold: int = 20, morph_kernel_size: int = 5):
        super().__init__(image_path)

        self.threshold = threshold
        self.morph_kernel_size = morph_kernel_size

    def predict(self):

        image = self.image

        r = image[0].astype(float)
        g = image[1].astype(float)
        b = image[2].astype(float)
        exg = 2 * g - r - b
        vegetation_mask = exg > self.threshold

        # Apply opening (erosion followed by dilation) to smooth mask boundaries
        structure = np.ones((self.morph_kernel_size, self.morph_kernel_size), dtype=bool)
        cleaned_mask = ndimage.binary_opening(vegetation_mask, structure=structure)

        self.mask = cleaned_mask

        return cleaned_mask