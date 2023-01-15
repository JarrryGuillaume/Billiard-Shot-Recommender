import matplotlib.pyplot as plt
import numpy as np

import physics_tools as pt
from config import ball_diameter, table_height, table_width, white_ball_size

list_of_balls = []
white_ball = []
list_of_holes = []


def generate_reachables_balls(white_ball, list_of_balls, list_of_holes):
    reachable_balls = []

    for ball_index, ball in enumerate(list_of_balls):
        for hole_index, hole in enumerate(list_of_holes):
            hit, trajectory = pt.hit_trajectory(ball, white_ball, hole, list_of_balls)
            hit = True
            out_vector = hole - ball
            shot_vector = trajectory * -1
            cosine_similarity = (
                out_vector
                @ shot_vector
                / (np.linalg.norm(out_vector) * np.linalg.norm(shot_vector))
            )
            if hit and cosine_similarity > 0:
                reachable_balls.append(
                    {
                        "ball_index": ball_index,
                        "hole_index": hole_index,
                        "out_vector": out_vector,
                        "shot_vector": shot_vector,
                    }
                )

    return reachable_balls


def transform_coordinates():
    balls, white, holes = [], [], []

    return balls, white, holes


def plot_game_state(
    table_holes_pos, balls_pos, white_ball_pos, recommended_shots, color=True
):
    fig, ax = plt.subplots()

    for hole_pos in table_holes_pos:
        ax.add_artist(
            plt.Circle(
                (hole_pos[1], hole_pos[0]),
                ball_diameter / 2,
                color="black",
                fill=True,
            )
        )

    for ball_pos in balls_pos:
        ax.add_artist(
            plt.Circle(
                (ball_pos[1], ball_pos[0]),
                ball_diameter / 2,
                color="red",
                fill=True,
            )
        )

    ax.add_artist(
        plt.Circle(
            (white_ball_pos[1], white_ball_pos[0]),
            white_ball_size / 2,
            color="blue",
            fill=True,
        )
    )

    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)
    ax.set_aspect("equal", adjustable="box")

    # Plot recommended shots
    for shot in recommended_shots:
        ball_pos = balls_pos[shot["ball_index"]]
        hole_pos = table_holes_pos[shot["hole_index"]]
        shot_vector = shot["shot_vector"]
        out_vector = shot["out_vector"]

        shot_vector = shot_vector / np.linalg.norm(shot_vector)
        out_vector = out_vector / np.linalg.norm(out_vector)

        dist_ball = np.linalg.norm(ball_pos - white_ball_pos)
        dist_hole = np.linalg.norm(hole_pos - ball_pos)

        cosine_similarity = (
            out_vector
            @ shot_vector
            / (np.linalg.norm(out_vector) * np.linalg.norm(shot_vector))
        )

        if color is True:
            cmap = plt.get_cmap("Reds")
        else:
            cmap = None

        ax.arrow(
            white_ball_pos[1],
            white_ball_pos[0],
            shot_vector[1] * dist_ball,
            shot_vector[0] * dist_ball,
            head_width=0.05,
            head_length=0.05,
            length_includes_head=True,
            color=cmap(cosine_similarity) if cmap is not None else "black",
        )

        ax.arrow(
            ball_pos[1],
            ball_pos[0],
            out_vector[1] * dist_hole,
            out_vector[0] * dist_hole,
            head_width=0.05,
            head_length=0.05,
            length_includes_head=True,
            color=cmap(cosine_similarity) if cmap is not None else "black",
        )

    if len(recommended_shots) > 0 and cmap is not None:
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
        cbar.ax.set_ylabel("Cosine similarity", rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

    fig.tight_layout()
    return fig, ax
