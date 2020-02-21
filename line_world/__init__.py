
from gym.envs.registration import register

register(
    id='LineWorld-v0',
    entry_point='line_world.envs:LineWorldEnv',
)
