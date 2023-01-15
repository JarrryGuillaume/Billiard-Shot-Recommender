# %%
import numpy as np

from config import table_height, table_width
from physics import generate_reachables_balls, plot_game_state

# %%

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
    [
        np.random.uniform(0, table_height, num_balls),
        np.random.uniform(0, table_width, num_balls),
    ],
    axis=-1,
)

white_ball_pos = np.array(
    [np.random.uniform(0, table_height), np.random.uniform(0, table_width)]
)

# %%
recommendations = generate_reachables_balls(white_ball_pos, balls_pos, table_holes_pos)
recommendations
# %%
plot_game_state(
    table_holes_pos,
    balls_pos,
    white_ball_pos,
    recommendations,
)

# %%

# %reload_ext autoreload
# %autoreload 2
# %%
