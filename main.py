import glob

import cv2 as cv
import numpy as np
from rembg import remove

import table_detection as td
from config import table_height, table_width
from physics import generate_reachables_balls, plot_game_state

datadir = "input/"
resultdir = "output/"


def remove_background(img):
    return remove(img.copy())


def get_transformed_pixel(x, y, h):
    p = np.array([[x], [y], [1]])
    p_new = np.matmul(h, p)
    m_new = p_new[0] / p_new[2]
    n_new = p_new[1] / p_new[2]
    return m_new[0], n_new[0]


def is_outside(x, y, size, margin=0):
    if -margin < x < size[0] + margin and -margin < y < size[1] - margin:
        return 0
    else:
        return 1


def draw_shots_in_image(img, shots, balls_pos, table_holes_pos, white_ball_pos):
    img = img.copy()
    img = cv.flip(img, 0)
    for shot in shots:
        ball_pos = balls_pos[shot["ball_index"]]
        hole_pos = table_holes_pos[shot["hole_index"]]
        shot_vector = shot["shot_vector"]
        out_vector = shot["out_vector"]

        shot_vector = shot_vector / np.linalg.norm(shot_vector)
        out_vector = out_vector / np.linalg.norm(out_vector)

        dist_ball = np.linalg.norm(ball_pos - white_ball_pos)
        dist_hole = np.linalg.norm(hole_pos - ball_pos)

        cv.arrowedLine(
            img,
            (
                int(white_ball_pos[1]),
                int(white_ball_pos[0]),
            ),
            (
                int(white_ball_pos[1]) + int(shot_vector[1] * dist_ball),
                int(white_ball_pos[0]) + int(shot_vector[0] * dist_ball),
            ),
            (0, 0, 255),
            2,
        )

        cv.arrowedLine(
            img,
            (
                int(ball_pos[1]),
                int(ball_pos[0]),
            ),
            (
                int(ball_pos[1])
                + int(
                    out_vector[1] * dist_hole,
                ),
                int(ball_pos[0])
                + int(
                    out_vector[0] * dist_hole,
                ),
            ),
            (0, 0, 255),
            2,
        )
    img = cv.flip(img, 0)
    return img


if __name__ == "__main__":
    for i, img_path in enumerate(glob.glob(datadir + "/*.png")):
        img = cv.imread(img_path)

        img_out = img.copy()

        img = remove_background(img)
        cv.imwrite(resultdir + f"removed_bg{i}.png", img)

        corners = td.detect_corners(img)

        img_table_detection = td.draw_edges(img, corners)
        cv.imwrite(resultdir + f"table{i}.png", img_table_detection)

        h, size = td.get_homography(corners)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_blur = cv.GaussianBlur(img_gray, (3, 3), 0.7)

        circles = cv.HoughCircles(
            img_blur,
            cv.HOUGH_GRADIENT,
            1,
            20,
            param1=15,
            param2=18,
            minRadius=0,
            maxRadius=20,
        )

        circles = np.uint16(np.around(circles))

        MOCK_X, MOCK_Y = 862, 339
        mock_circle = [[[MOCK_X, MOCK_Y, 9]]]
        circles = np.concatenate([circles, mock_circle], axis=1)

        highest_avg_color = 0

        white_ball = None
        balls = []
        for circle in circles[0, :]:
            x, y, r = circle

            x_new, y_new = get_transformed_pixel(x, y, h)

            if (not (x == MOCK_X and y == MOCK_Y)) and (
                is_outside(x_new, y_new, size) == 1
            ):
                continue
            # if is_outside(x_new, y_new, size) == 1:
            #     continue

            ball = (y_new, x_new)
            balls.append(ball)

            mask = np.zeros(img.shape[:2], np.uint8)
            cv.circle(mask, (int(x), int(y)), int(r), 255, -1)
            avg_color = cv.mean(img_gray, mask=mask)
            if avg_color[0] > highest_avg_color:
                highest_avg_color = avg_color[0]
                whitest_circle = circle

            # draw the outer circle
            cv.circle(img_out, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(img_out, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        if len(balls) == 0:
            raise Exception("No balls detected")

        # Draw the whitest circle on the image
        if whitest_circle is not None:
            x, y, r = whitest_circle
            cv.circle(img_out, (int(x), int(y)), int(r), (255, 0, 0), 2)
            x_new, y_new = get_transformed_pixel(x, y, h)
            white_ball = (y_new, x_new)

        cv.imwrite(resultdir + f"white_ball_detection{i}.png", img_out)

        table_holes = np.array(
            [
                [0, 0],
                [0, table_width],
                [table_height, 0],
                [table_height, table_width],
                [0, table_width / 2],
                [table_height, table_width / 2],
            ]
        )
        balls = np.array(balls)
        white_ball = np.array(white_ball)
        balls = balls[:, :2]
        white_ball = white_ball[:2]

        balls[:, 0] = table_height - balls[:, 0]
        white_ball[0] = table_height - white_ball[0]

        # recommendations = generate_reachables_balls(white_ball, balls, table_holes)
        recommendations = []
        # recommendations = np.random.choice(recommendations, 3)

        fig, ax = plot_game_state(
            table_holes,
            balls,
            white_ball,
            recommendations,
        )
        fig.savefig(resultdir + f"game_state{i}.png")

        img_warped = cv.warpPerspective(img, h, size)
        cv.imwrite(resultdir + f"warped{i}.png", img_warped)

        img_shots = draw_shots_in_image(
            img_warped, recommendations, balls, table_holes, white_ball
        )
        cv.imwrite(resultdir + f"shots{i}.png", img_shots)
