#%%
import matplotlib.pyplot as plt
import numpy as np

table_height = 2
table_width = 1

table_holes_pos = np.array(
    [
        [0, 0],
        [0, table_width],
        [table_height / 2, 0],
        [table_height / 2, table_width],
        [table_height, 0],
        [table_height, table_width],
    ]
)

# %%
num_balls = 8
balls_pos = np.stack(
    [np.random.uniform(0, table_height, num_balls), np.random.uniform(0, table_width, num_balls)], axis=-1
)

white_ball_pos = np.array([np.random.uniform(0, table_height), np.random.uniform(0, table_width)])

# %%
def plot_game_state(table_holes_pos, balls_pos, white_ball_pos, recommended_shots):
    fig, ax = plt.subplots()

    ax.scatter(table_holes_pos[:, 1], table_holes_pos[:, 0], c="black")
    ax.scatter(balls_pos[:, 1], balls_pos[:, 0], c="red")

    ax.scatter(white_ball_pos[1], white_ball_pos[0], c="blue")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal", adjustable="box")

    # Plot recommended shots
    for shot in recommended_shots:
        ball_pos = balls_pos[shot["ball_index"]]
        hole_pos = table_holes_pos[shot["hole_index"]]
        out_vector = shot["out_vector"]

        ax.arrow(
            white_ball_pos[1],
            white_ball_pos[0],
            # TODO: use actual shot recommendations
            # shot_vector[1],
            # shot_vector[0],
            ball_pos[1] - white_ball_pos[1],
            ball_pos[0] - white_ball_pos[0],
            head_width=0.05,
            head_length=0.05,
            length_includes_head=True,
            fc="k",
            ec="k",
        )
        ax.arrow(
            ball_pos[1],
            ball_pos[0],
            # TODO: use actual shot recommendations
            # out_vector[1],
            # out_vector[0],
            hole_pos[1] - ball_pos[1],
            hole_pos[0] - ball_pos[0],
            head_width=0.05,
            head_length=0.05,
            length_includes_head=True,
            fc="k",
            ec="k",
        )

    return fig, ax


# %%


def get_shot_recommendations(
    table_holes_pos, balls_pos, white_ball_pos, ball_radius=0.05
):
    # Use billard shock equations to calculate vectors

    shot_recommendations = []
    for ball_index, ball_pos in enumerate(balls_pos):
        for hole_index, hole_pos in enumerate(table_holes_pos):
            hole_pos = table_holes_pos[hole_index]
            # Calculate shot vector
            shot_vector = ball_pos - white_ball_pos
            shot_vector = shot_vector / np.linalg.norm(shot_vector)

            # Calculate out vector
            out_vector = hole_pos - ball_pos
            out_vector = out_vector / np.linalg.norm(out_vector)

            cos_distance = np.dot(shot_vector, out_vector)
            if cos_distance > 0.5:
                shot_recommendations.append(
                    {
                        "ball_index": ball_index,
                        "hole_index": hole_index,
                        "shot_vector": shot_vector,
                        "out_vector": out_vector,
                    }
                )

    return shot_recommendations


#%%
recommended_shots = get_shot_recommendations(table_holes_pos, balls_pos, white_ball_pos)

plot_game_state(table_holes_pos, balls_pos, white_ball_pos, recommended_shots)

# %%
