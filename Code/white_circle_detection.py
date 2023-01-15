#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image and convert it to grayscale
image = cv2.imread("./input/table_top.png")

width, height = image.shape[:2]
#%%
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use the HoughCircles method to detect circles in the image
circles = cv2.HoughCircles(
    image, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=20, minRadius=0, maxRadius=30
)

#%%
# Iterate over the detected circles and measure the average color of each circle
whitest_circle = None
highest_avg_color = 0

margin = 50
for circle in circles[0]:
    x, y, r = circle

    # TODO: use detected table instead of margin
    if x < margin or x > width - margin or y < margin or y > height - margin:
        continue

    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    avg_color = cv2.mean(image, mask=mask)
    if avg_color[0] > highest_avg_color:
        highest_avg_color = avg_color[0]
        whitest_circle = circle

# Draw the whitest circle on the image
if whitest_circle is not None:
    x, y, r = whitest_circle
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# %%
