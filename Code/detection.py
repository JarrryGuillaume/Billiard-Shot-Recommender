#%%
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

output_dir = "output"

#%%
def load_images(dir):
    images = []
    for filename in os.listdir(dir):
        img = cv.imread(os.path.join(dir, filename))
        if img is not None:
            images.append(img)
    return images


imgs = load_images("input")

# %%
def detect_circles(img_in):
    img = cv.medianBlur(img_in, 3)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(
        img, cv.HOUGH_GRADIENT, 1, 20, param1=30, param2=20, minRadius=0, maxRadius=30
    )
    return circles


def detect_lines(img_in):
    img = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    return lines


def draw_circles(img_in, circles):
    img_out = img_in.copy()
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img_out, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img_out, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img_out


def draw_lines(img_in, lines):
    img_out = img_in.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_out


def plot_images(imgs, name):
    cols = min(len(imgs), 3)
    fig, axes = plt.subplots(
        figsize=(10, 5), nrows=np.ceil(len(imgs) / cols).astype(int), ncols=cols
    )

    axes_list = [axes] if cols == 1 else list(axes.flat)

    for ax in axes_list:
        ax.set_axis_off()
    for ax, img in zip(axes_list, imgs):
        ax.set_axis_on()
        ax.imshow(img)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(f"output/{name}.png", dpi=300, bbox_inches="tight")


# %%
circles = [detect_circles(img) for img in imgs]
imgs_out = [draw_circles(img, circles) for img, circles in zip(imgs, circles)]
plot_images(imgs_out, "circles")
# %%
lines = [detect_lines(img) for img in imgs]
imgs_out = [draw_lines(img, lines) for img, lines in zip(imgs, lines)]
plot_images(imgs_out, "lines")

# %%
