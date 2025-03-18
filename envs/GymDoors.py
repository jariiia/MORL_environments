from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, RenderFrame, ActType

from Doors import Environment


class GymEnvironment(Environment, gym.Env):
    """
    A Gymnasium environment for the Doors problem.
    """

    metadata = {'render.modes': ['human']}

    action2string = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
        4: "toggle_door",
    }

    sting2action = {
        "up": 0,
        "right": 1,
        "down": 2,
        "left": 3,
        "toggle_door": 4,
    }

    def __init__(self, mode:str="scalarised", we:float=3.0):
        super(GymEnvironment, self).__init__()
        super().__init__()
        self.mode = mode
        self.we = we
        self.env_init()
        self.step_count = 0
        self.max_steps = 50

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        obs = self.env_start()
        self.step_count = 0
        return obs, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.visualise_environment()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not isinstance(action, str):
            action = self.action2string[action]

        rewards, observation = self.env_step(action)
        observation = self.prep_obs(observation)
        tr = self.step_count >= self.max_steps
        tm = self.is_terminal()

        r0 = 0 if not self.is_terminal() else self.DOORS_OPEN_PENALTY*self.doors_open_count
        r1 = -1 if not self.is_terminal() else 50

        if self.mode == "scalarised":
            reward = float(r0*self.we + r1)
        else:
            reward = np.array([r0, r1], dtype=np.float32)
        return observation, reward, tm, tr, {}


    def prep_obs(self, obs):
        return np.array(obs,dtype=np.float32)/np.array([14-1, 2-1, 2-1],dtype=np.float32)

if __name__ == '__main__':

    e = GymEnvironment(mode="vector")

    # testing trajectory around doors
    actions = ['right','right','right','down','down','down','down','left','left','left']

    # testing trajectory through doors opening doors and leaving them open
    # actions = ['down', 'toggle_door', 'down', 'down', 'toggle_door', 'down', 'down']

    # testing trajectory through doors opening doors and closing them
    # actions = ['down', 'toggle_door', 'down', 'toggle_door', 'down','toggle_door', 'down', 'toggle_door', 'down']


    observation, _ = e.reset()
    print('Observation', observation)
    rewards = np.zeros(2, dtype=np.float32)
    for a in actions:
        observation, r, tm, tr, _ = e.step(a)
        e.render()
        rewards += r
        print('Action:', a)
        print('Rewards:', r)
        print('Observation', observation)

    print("\nIs terminal?", e.is_terminal())
    print("\nValue vector?", rewards)