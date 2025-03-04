from typing import SupportsFloat, Any

from gymnasium.core import ActType, ObsType, RenderFrame

from Breakable_bottles import BreakableBottles
import gymnasium as gym


class GymBreakableBottles(BreakableBottles, gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GymBreakableBottles, self).__init__()
        super().__init__()

        self.env_start()
        self.step_count = 0
        self.max_steps = 100

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        rewards, observation = self.env_step(action)
        tr = self.step_count >= self.max_steps
        tm = self.is_terminal()
        return observation, rewards, tm, tr, {},

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.visualise_environment()

    def reset(self) -> ObsType:
        self.step_count = 0
        self.env_clean_up()
        self.env_init()
        return self.env_start()
        pass

if __name__ == "__main__":
    actions = ['pick_up_bottle', 'pick_up_bottle', 'right', 'right', 'right', 'right', 'left', 'left', 'left', 'left',
               'pick_up_bottle', 'right', 'right', 'right', 'right']

    # testing trajectory picking up two bottles
    # actions = ['pick_up_bottle','pick_up_bottle', 'right', 'right', 'right', 'right']
    e = GymBreakableBottles()
    e.visualise_environment()
    for a in actions:
        rewards, observation = e.env_step(a)
        e.visualise_environment()
        """print('Action:', a)
        print('Rewards:', rewards)
        print('Observation', observation)"""

    print("\nIs terminal?", e.is_terminal())

