import math

import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

from config import table_height, table_width


def Distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def sort_rho(edge):
    return edge[0]


def sort_by_average_rho(edge):
    sum_rho = 0
    for line in edge:
        sum_rho += line[0]
    average_rho = sum_rho / len(edge)
    return average_rho


def convert_to_positive_rho(lines):
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        if rho < 0:
            line[0][0] = -line[0][0]
            line[0][1] = theta - np.pi
    return lines


def split_edges_by_rho(edges):
    edges = np.array(edges)
    clustering_rho1 = KMeans(n_clusters=2).fit(edges[:, 0].reshape(-1, 1))
    labels = clustering_rho1.labels_
    edges_split = [[] for _ in range(2)]
    for i in range(len(edges)):
        edges_split[labels[i]].append([edges[i][0], edges[i][1]])
    edges_split.sort(key=sort_by_average_rho)
    return edges_split


def get_edges(lines):
    clustering_theta = KMeans(n_clusters=2, random_state=0).fit(
        lines[:, 1].reshape(-1, 1)
    )
    labels_theta = clustering_theta.labels_
    edges_theta1 = []
    edges_theta2 = []
    for i in range(len(lines)):
        if labels_theta[i] == 0:
            edges_theta1.append([lines[i][0], lines[i][1]])
        else:
            edges_theta2.append([lines[i][0], lines[i][1]])
    edges_rho1 = split_edges_by_rho(edges_theta1)
    edges_rho2 = split_edges_by_rho(edges_theta2)
    edges = [[] for _ in range(4)]
    if clustering_theta.cluster_centers_[0] < clustering_theta.cluster_centers_[1]:
        edges[0] = edges_rho1[0]
        edges[1] = edges_rho2[0]
        edges[2] = edges_rho1[1]
        edges[3] = edges_rho2[1]
    else:
        edges[0] = edges_rho2[0]
        edges[1] = edges_rho1[0]
        edges[2] = edges_rho2[1]
        edges[3] = edges_rho1[1]
    return edges


def getInnerEdges(edges):
    inner_edges = np.zeros((4, 2))
    for i in range(2):
        edges[i].sort(key=sort_rho)
        edges[3 - i].sort(reverse=True, key=sort_rho)
    for i in range(4):
        inner_edges[i] = edges[i][-1]
    return inner_edges


def get_line_parameters(edge):
    theta = edge[1]
    a = math.cos(theta)
    b = math.sin(theta)
    c = edge[0]
    return a, b, c


def get_intersection(line1, line2):
    a1 = line1[0]
    b1 = line1[1]
    c1 = line1[2]
    a2 = line2[0]
    b2 = line2[1]
    c2 = line2[2]
    determinant = a1 * b2 - a2 * b1
    x = round((b2 * c1 - b1 * c2) / determinant)
    y = round((a1 * c2 - a2 * c1) / determinant)
    return x, y


def get_corners(edges):
    edges_cartesian = np.zeros((4, 3))
    for i in range(4):
        edges_cartesian[i] = get_line_parameters(edges[i])
    corner1 = get_intersection(edges_cartesian[0], edges_cartesian[1])
    corner2 = get_intersection(edges_cartesian[1], edges_cartesian[2])
    corner3 = get_intersection(edges_cartesian[2], edges_cartesian[3])
    corner4 = get_intersection(edges_cartesian[3], edges_cartesian[0])
    return np.array([corner1, corner2, corner3, corner4])


def get_corresponding_points(corners):
    if Distance(corners[0], corners[1]) > Distance(corners[0], corners[3]):
        width = table_width
        height = table_height
    else:
        width = table_height
        height = table_width
    width = table_width
    height = table_height

    # NOTE: hard-coded for cases where homography stretches the table in wrong direction
    flipped = True
    if not flipped:
        p1 = [0, 0]
        p2 = [width, 0]
        p3 = [width, height]
        p4 = [0, height]
    else:
        p4 = [0, 0]
        p1 = [width, 0]
        p2 = [width, height]
        p3 = [0, height]
    return np.array([p1, p2, p3, p4]), (width, height)


def detect_corners(img):
    img_blur = cv.GaussianBlur(img, (3, 3), 0.6)
    img_edges = cv.Canny(img_blur, 40, 170, 3)
    lines = cv.HoughLines(img_edges, 1, np.pi / 180, 170, None, 0, 0)
    if lines is not None:
        lines = convert_to_positive_rho(lines)
        edges = get_edges(lines[:, 0, :])
        edges = getInnerEdges(edges)
    return get_corners(edges)


def get_homography(corners):
    corresponding_points, size = get_corresponding_points(corners)
    h, status = cv.findHomography(corners, corresponding_points)
    return h, size


def draw_edges(img_in, corners):
    img_out = img_in.copy()
    for i in range(4):
        pt1 = (corners[i][0], corners[i][1])
        pt2 = (corners[(i + 1) % 4][0], corners[(i + 1) % 4][1])
        cv.line(img_out, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
    return img_out


def warp_image(img, h, size):
    return cv.warpPerspective(img, h, size)
