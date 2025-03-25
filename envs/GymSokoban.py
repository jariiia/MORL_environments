from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gym.core import ActType, ObsType, RenderFrame
from numpy.random import default_rng

from Sokoban import Environment as SokobanEnv

class GymSokoban(SokobanEnv, gym.Env):
    """
    This class wraps the Sokoban environment for use with Gymnasium.
    It inherits from the Sokoban environment class and implements the necessary methods
    to conform to the OpenAI Gym API.
    """

    metadata = {'render.modes': ['human']}

    action2string = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    sting2action = {
        "up": 0,
        "right": 1,
        "down": 2,
        "left": 3,
    }

    def __init__(self, mode: str = "scalarised", we: float = 3.0, normalised_obs: bool = True):
        super(GymSokoban, self).__init__()
        super().__init__()
        self.normalised_obs = normalised_obs
        self.mode = mode
        self.we = we
        self.env_start()
        self.step_count = 0
        self.max_steps = 50

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32) if self.normalised_obs else gym.spaces.MultiDiscrete([11, 11])
        self.action_space = gym.spaces.Discrete(4)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not isinstance(action, str):
            action = self.action2string[action]
        rewards, observation = self.env_step(action)
        self.step_count += 1
        tr = self.step_count >= self.max_steps
        tm = self.is_terminal()

        r0 = 0 if not tm else 50 + self.BOX_PENALTY[self.box_location]
        r1 = -1 if not tm else 50

        if self.mode == "scalarised":
            rewards = r0 * self.we + r1
        else:
            rewards = np.array([r0, r1]).astype(np.float32)

        return self.prep_obs(observation), rewards, tm, tr, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.visualise_environment()

    def prep_obs(self, obs):
        if self.normalised_obs:
            return np.array(obs, dtype=np.float32) / np.array([11-1, 11-1], dtype=np.float32)
        else:
            return np.array(obs, dtype=np.float32)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = default_rng(seed=seed)
        self.step_count = 0
        self.env_clean_up()
        self.env_init()
        return self.prep_obs(self.env_start()), {}

if __name__ == '__main__':

    e = GymSokoban(mode="vector")
    e.render()

    # testing trajectory starting from the left SUBOPTIMAL
    actions = ['left', 'down', 'right', 'down', 'right', 'down', 'right']

    # testing trajectory pushing box down from the starting cell WORST
    # actions = ['down', 'right', 'down', 'down', 'right']

    # testing trajectory leaving the box at its original position OPTIMAL
    # actions = ['left', 'down', 'right', 'down', 'right', 'right', 'up', 'left', 'down', 'down', 'right']

    obs, _ = e.reset(seed=0)
    e.render()
    rew = np.array([0., 0.])
    for a in actions:
        obs, rewards, tm, tr, info = e.step(a)
        rew += rewards
        e.render()
        print('Action:', a)
        print('Rewards:', rewards)
        print('Observation', obs)
    print('Rewards:', rew)
    print("\nIs terminal?", e.is_terminal())

