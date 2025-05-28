import cv2
import numpy as np
import matplotlib.pyplot as plt
# from excess_green import mask2geojson, load_image

image_path = 'samgeo_tests/satellite.tif'
# image_path = 'samgeo_tests/satellite_sanchisguarner.tif'
image_path = '../samgeo_tests/satellite_sanvicent.tif'

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# White mask: high V, low S
white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
# Optional: Yellow mask (if lines are yellow in some countries)
yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))

lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

masked = cv2.bitwise_and(gray, gray, mask=lane_mask)

# Canny edge detection
edges = cv2.Canny(masked, threshold1=50, threshold2=150, apertureSize=3)

# Morphological operations to clean up
# Dilate to close gaps, then erode to thin
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edges, kernel, iterations=1)
cleaned = cv2.erode(dilated, kernel, iterations=1)

thinned = cleaned

# Filter short segments using contour area and get thin lines
contours, _ = cv2.findContours(thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
lane_mask = np.zeros_like(gray)

for cnt in contours:
    length = cv2.arcLength(cnt, closed=False)
    if length > 20:  # Keep long segments only
        cv2.drawContours(lane_mask, [cnt], -1, 255, thickness=1)

# lane_mask = (lane_mask > 0).astype("uint8")

# _, transform, crs = load_image(image_path)
# mask2geojson(lane_mask, transform, crs, file_name="roads_sanvicent.geojson")

# --- 6. Show results ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# plt.title("Input")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.subplot(1, 4, 2)
# # plt.title("Edges")
# plt.imshow(edges, cmap="gray")
plt.subplot(1, 2, 2)
# plt.title("Final Lane Mask")
plt.imshow(thinned, cmap="gray")
# plt.subplot(1, 3, 3)
# plt.title("Final Lane Mask")
plt.imshow(lane_mask, cmap="gray")
plt.tight_layout()
plt.savefig("lanes_detected.jpg")
plt.show()
