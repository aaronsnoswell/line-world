"""Test the line_world environment"""


from line_world.envs.line_world_env import LineWorldEnv


def test_gym_spec():
    """Verify the LineWorld env against the gym spec
    
    See https://github.com/openai/gym/blob/master/gym/core.py for more
    information
    
    """
    
    from stable_baselines.common.env_checker import check_env
    
    print("Testing LineWorldEnv() matches OpenAI gym.Env Spec")
    env = LineWorldEnv()
    check_env(
        env,
        warn=True,
        skip_render_check=False
    )
    

if __name__ == '__main__':
    test_gym_spec()
    print("Done")


