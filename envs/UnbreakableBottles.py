from typing import SupportsFloat, Any

import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from numpy.random import default_rng

from Unbreakable_bottles import UnbreakableBottles
import gymnasium as gym


class GymUnbreakableBottles(UnbreakableBottles, gym.Env):
    metadata = {'render.modes': ['human']}

    action2string = {
        0: "left",
        1: "right",
        2: "pick_up_bottle"
    }

    sting2action = {
        "left": 0,
        "right": 1,
        "pick_up_bottle": 2
    }

    def __init__(self, mode: str = "scalarised", we: float = 3.0, normalised_obs: bool = True):
        super(GymUnbreakableBottles, self).__init__()
        super().__init__()
        self.mode = mode
        self.we = we
        self.env_start()
        self.step_count = 0
        self.max_steps = 50
        self.normalised_obs = normalised_obs
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) if self.normalised_obs else gym.spaces.MultiDiscrete([5, 3, 2, 2**3])
        self.action_space = gym.spaces.Discrete(3)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not isinstance(action, str):
            action = self.action2string[action]
        rewards, observation = self.env_step(action)
        self.step_count += 1
        tr = self.step_count >= self.max_steps
        tm = self.is_terminal()

        r0 = 0 if not (tm or tr) else -50*self.bottles_on_floor
        r1 = -1+self.bottles_delivered_this_step*25

        if self.mode == "scalarised":
            rewards =  r0*self.we + r1
        else:
            rewards = np.array([r0, r1]).astype(np.float32)

        return self.prep_obs(observation), rewards, tm, tr, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.visualise_environment()

    def prep_obs(self, obs):
        obs = np.array([obs[0], obs[1], obs[3], 2**obs[2][0] + 2**obs[2][1] + 2**obs[2][2]] )
        if self.normalised_obs:
            return obs / np.array([5 - 1, 3 - 1, 2 - 1, 2 ** 3 - 1])
        return obs


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

if __name__ == "__main__":
    actions = ['pick_up_bottle','pick_up_bottle', 'right', 'right', 'right','pick_up_bottle', 'right']

    e = GymUnbreakableBottles(mode="vector")
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

