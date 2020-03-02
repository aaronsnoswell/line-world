
import gym
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


class LineWorldEnv(gym.Env):
    """A simple multi-modal RL Environment"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            *,
            stochastic=True,
            mode_weights=np.array([0.5, 0.5]),
            mode_offset=0.0,
            mode_separation=10.0,
            mode_std=2.0,
            episode_length=None
        ):
        """C-tor

        args:
            stochastic (bool): If true, add Normal action noise
            mode_offset (float): Offset in state-space of the two modes
            mode_separation (float): Distance between the two reward modes
            mode_std (float): Std deviation of the reward gaussians
            episode_length (int): Number of time steps in an episode. If None,
                the problem will be infinite-horizon.
        """

        assert len(mode_weights) == 2,\
            "len(mode_weights) must be == 2, is {}".format(len(mode_weights))

        if sum(mode_weights) != 1.0:
            warnings.warn("sum(mode_weights) != 1.0, normalizing")
            mode_weights = np.array(mode_weights) / sum(mode_weights)

        self._starting_state_distribution = norm()
        self._stochastic = stochastic
        self._dynamics_noise_distribution = (
            norm(0.0, 0.2 * float(stochastic))
        )

        self._mode_offset = mode_offset
        self._mode_separation = mode_separation
        self._peak_value = 1 / np.sqrt(2 * np.pi * mode_std ** 2)
        self._reward1 = lambda s:\
            float(norm(
                mode_offset + mode_separation / 2.0,
                mode_std
            ).pdf(s))
        self._reward2 = lambda s:\
            float(norm(
                mode_offset - mode_separation / 2.0,
                mode_std
            ).pdf(s))
        self._mode_weights = mode_weights

        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Box(
            np.array([-1]),
            np.array([1])
        )
        self.observation_space = gym.spaces.Box(
            np.array([-np.inf]),
            np.array([np.inf])
        )

        self._episode_length = episode_length

        # Viewer for human visualization
        self.viewer = None

        self._state = 0.0
        self._timestep = 0

        self.reset()

    def step(self, action):
        """Step the environment

        Args:
            action (float): Action to take

        Returns:
            (float): Observation
            (float): Reward
            (bool): True if the episode is finished
            (dict): Extra information dictionary

        """
        
        # Clamp action to valid range
        if not self.action_space.contains(action):
            # warnings.warn("Action ({}) is outside action_space ({})".format(
            #     action,
            #     self.action_space
            # ))
            action = np.clip(
                action,
                self.action_space.low,
                self.action_space.high
            )

        # Apply action
        self._state += action
        self._state += self._dynamics_noise_distribution.rvs()

        self._timestep += 1

        ob = self._state
        reward = self._get_reward(self._state)
        if self._episode_length is None:
            episode_over = False
        else:
            episode_over = self._timestep >= self._episode_length
        info = {}
        return ob, reward, episode_over, info

    def reset(self):
        """Reset the environment"""
        self._state = np.array([self._starting_state_distribution.rvs()])
        self._timestep = 0
        return self._state

    def render(self, mode='human', close=False):
        """Render the task for human consumption

        Args:
            mode (str): Only supported options are 'human' or 'rgb_array'
            close (bool): Close the window this call

        Returns:
            (numpy array): Rendered image of the scene
        """
        screen_width = 600
        screen_height = 400
        x_scale = 15
        color_default = np.array([0.7, 0.5, 0.3])
        color_highlight = np.array([0.0, 0.5, 0.3])

        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        # First time set up
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Draw the reward distribution
            _x = np.linspace(-20, 20, 200)
            _r = [self._get_reward(x) for x in _x]
            reward_curve = rendering.PolyLine(list(zip(_x, _r)), close=False)
            reward_curve.set_color(*color_highlight)
            reward_curve.add_attr(rendering.Transform(
                translation=[screen_width / 2, screen_height / 3],
                scale=[x_scale, screen_height / 2]
            ))
            self.viewer.add_geom(reward_curve)

            # Draw the line
            line = rendering.Line((-1, 0), (1, 0))
            self._v_base_tform = rendering.Transform(
                translation=[screen_width / 2, screen_height / 3],
                scale=[screen_width, screen_height]
            )
            line.add_attr(self._v_base_tform)
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

            # Draw the state
            self._v_state = rendering.make_circle(10)
            self._v_state_tform = rendering.Transform()
            self._v_state.add_attr(self._v_state_tform)
            self.viewer.add_geom(self._v_state)

        # Update the position of the dot
        self._v_state_tform.set_translation(
            screen_width / 2 + self._state * x_scale,
            screen_height / 3
        )

        # Set color of the dot based on current reward
        reward = self._get_reward(self._state)
        blend_coeff = reward ** 4
        self._v_state.set_color(
            *(
                blend_coeff * color_highlight +
                (1.0 - blend_coeff) * color_default
            )
        )

        return self.viewer.render(
            return_rgb_array=(mode == 'rgb_array')
        )

    def _get_reward(self, s):
        """Get reward

        Args:
            s (float): State to sample reward from

        Returns:
            (float): Reward
        """
        # We normalize the reward so it is on the range [0, 1]
        return float(
            self._mode_weights @ [self._reward1(s), self._reward2(s)] /
            self._peak_value / np.max(self._mode_weights)
        )

    def _starting_prob(self, s):
        """The probability of starting in state s

        Args:
            s (float): State

        Returns:
            (float): Probability of starting in state s
        """
        return self._starting_state_distribution.pdf(s)

    def _transition_prob(self, s1, a, s2):
        """The probability of reaching s2 given s1, a

        Args:
            s1 (float): Starting state
            a (float): Action
            s2 (float): Ending state

        Returns:
            (float): p(s2 | s1, a)
        """
        if self._stochsatic:
            return self._dynamics_noise_distribution.pdf(s2 - (s1 + a))
        else:
            return s2 == s1 + a

    def _trajectory_prob(self, trajectory):
        """The un-normalized probability of observing a state-action trajectory

        Args:
            trajectory (list): List of (s, a) pairs, terminated by (s, None)

        Returns:
            (float): Probability of observing this trajectory under the
                transition dynamics
        """

        # Accumulate probabilities in log space for numerical stability
        logprob = np.log(self._starting_prob(trajectory[0][0]))
        logprob += np.sum([
            np.log(self._transition_prob(s1, a, s2))
            for (s1, a), (s2, _) in zip(trajectory[:-1], trajectory[1:])
        ])

        return np.exp(logprob)

    def _opt_pol(self, *, s=None):
        """The optimal deterministic policy for symmetric problem definitions

        Args:
            s (float): Current state. If None, internal state is used.

        Returns:
            (float): Current action
        """

        if s is None:
            s = self._state

        rel_pos = s - self._mode_offset
        r1_pos = -self._mode_separation / 2
        r2_pos = -r1_pos

        r1_a = np.clip(r1_pos - rel_pos, -1, 1)
        r2_a = np.clip(r2_pos - rel_pos, -1, 1)

        if rel_pos < 0:
            # Target left peak
            return r1_a
        else:
            # Target right peak
            return r2_a

    def _viz_policy(self, policy, *, n_steps=100):
        """Visualize a policy

        Renders a rollout of the policy, and plots the policy in State-Action
        space

        Args:
            policy (function): Policy taking an observation and giving an action
            n_steps (function): Number of steps to roll the policy out for
        """

        obs = self.reset()

        # Roll out the policy
        total_reward = 0
        for n in range(n_steps):
            self.render()
            action = policy(obs)
            obs, reward, done, info = self.step(action)
            total_reward += reward
            if done: break

        self.render(close=True)
        print("Total reward = {:.2f}".format(total_reward))
        print("Average reward over {} steps = {:.2f}".format(
            n,
            total_reward / n
        ))

        # Render policy in State-Action space
        states = np.linspace(-10, 10, 400)
        actions = [policy(np.array([s])) for s in states]
        plt.figure()
        plt.plot(states, actions)
        plt.grid()
        plt.ylim(-1.1, 1.1)
        plt.title("LineWorld Policy")
        plt.xlabel("State")
        plt.ylabel("Action")
        plt.show()


def stable_baselines_demo(
        *,
        save_path="PPO2_LineWorld",
        n_envs=8,
        total_timesteps=100000,
        **kwargs
    ):
    """Train an agent using stable_baselines

    This uses PPO2 with a vectorized environment, as such it requires
    stable_baselines >= 2.9.0 to be installed.

    Args:
        save_path (str): Save the trained policy to this path
        n_env (int): Number of processes to use with PPO
        total_timesteps (int): Number of training timesteps

        kwargs (dict): Additional keyword args for stable_baselines.PPO2()
    """

    # Train a stable-baselines agent
    import os
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines import PPO2

    # We train with an episodic version of the task
    env = make_vec_env(
        lambda: LineWorldEnv(episode_length=50),
        n_envs=n_envs
    )

    if os.path.exists("{}.zip".format(save_path)):
        print("Loading pre-trained model from {}".format(save_path))
        model = PPO2.load(save_path, env, **kwargs)
    else:
        print("Training new PPO2 model")
        model = PPO2(
            MlpPolicy,
            env,
            verbose=1,
            **kwargs
        )

    # Train
    model.learn(
        total_timesteps=total_timesteps
    )

    if save_path is not None:
        # Save model
        print("Saving trained model to {}".format(save_path))
        model.save(save_path)

        # Load the model fresh to demonstrate save/load functionality
        del model
        model = PPO2.load(save_path)

    # Visualize the policy
    env = LineWorldEnv()
    env._viz_policy(
        lambda s: model.predict(s)[0]
    )


def demo():
    """Demo function"""

    env = LineWorldEnv()

    # Visualise a policy
    env._viz_policy(
        #lambda s: env.action_space.sample(),    # Uniform random policy
        #lambda s: 0.0,                          # Zero policy
        lambda s: env._opt_pol(s=s),            # Heuristic optimal policy
    )


if __name__ == '__main__':
    # Train a PPO2 agent
    #stable_baselines_demo()

    # Visiualize a policy
    demo()

