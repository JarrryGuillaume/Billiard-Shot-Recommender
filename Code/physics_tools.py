import numpy as np

import config


def scalar_opposite(X):
    return np.array([X[1], -X[0]]) / np.linalg.norm(X)


def hit_trajectory(ball, white_ball, hole, list_of_balls):
    # AC refers to "after collision", BC to "before collision"
    to_hole = line(ball, hole)

    # col stands for "collision"
    white_col_center = ball + config.ball_diameter * to_hole
    trajectory = line(white_ball, white_col_center)

    hit = check_hit(trajectory, white_ball, list_of_balls)
    return hit, trajectory


def check_hit(trajectory, white_ball, list_of_balls):
    """check if the trajectory of the white ball is not blocked by another ball"""
    for ball in list_of_balls:
        if cross_path(ball, trajectory, white_ball):
            return False
    return True


def cross_path(ball, trajectory, white_ball):
    """check if a given ball is on the way of the white ball"""
    normal = scalar_opposite(trajectory)
    b = np.dot(normal, white_ball)
    eps = config.ball_diameter
    if abs(np.dot(ball, normal)) < b + eps and abs(np.dot(ball, normal)) > b - eps:
        return True
    return False


def line(X, Y):
    """return the 2D directonary vector of the line between to point"""
    vector = np.array([X[0] - Y[0], X[1] - Y[1]])
    return vector / np.linalg.norm(vector)
