from typing import SupportsFloat, Any

import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame

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

    def __init__(self, mode: str = "scalarised", we: float = 3.0):
        super(GymUnbreakableBottles, self).__init__()
        super().__init__()
        self.mode = mode
        self.we = we
        self.env_start()
        self.step_count = 0
        self.max_steps = 50

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if isinstance(action, int):
            action = self.action2string[action]
        rewards, observation = self.env_step(action)
        tr = self.step_count >= self.max_steps
        tm = self.is_terminal()
        if self.mode == "scalarised":
            rewards = (rewards["PERFORMANCE_REWARD"]) + self.we * rewards["IMPACT_REWARD"]
        else:
            rewards = np.array([rewards["IMPACT_REWARD"], rewards["PERFORMANCE_REWARD"]])
        return observation, rewards, tm, tr, {},

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.visualise_environment()

    def reset(self) -> ObsType:
        self.step_count = 0
        self.env_clean_up()
        self.env_init()
        return self.env_start(), {}
        pass

if __name__ == '__main__':

    e = UnbreakableBottles()

    # testing trajectory picking up two bottles and picking bottles that dropped on the way back
    actions = ['pick_up_bottle','pick_up_bottle', 'right', 'right', 'right', 'right','left', 'pick_up_bottle', 'left', 'pick_up_bottle', 'left', 'pick_up_bottle', 'right', 'right', 'right', 'right']

    # testing trajectory picking up two bottles
    # actions = ['pick_up_bottle','pick_up_bottle', 'right', 'right', 'right', 'right']


    e.env_start()

    for a in actions:
        rewards, observation = e.env_step(a)
        e.visualise_environment()
        print('Action:', a)
        print('Rewards:', rewards)
        print('Observation', observation)
        print("aaaaaaaaaaa")
        print(e.get_state())
        print("eeeeeeeee")

    print("\nIs terminal?", e.is_terminal())