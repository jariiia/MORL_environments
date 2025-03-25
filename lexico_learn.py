"""
In this script we will learn the reference policy for the given environment using the LPPO algorithm and the ExperimentRunner class from the libraries:
- https://github.com/maymac00/mo-stable-baselines3
- https://github.com/maymac00/experiment_runner
respectively.

Both can be installed using pip:
pip install git+https://github.com/maymac00/experiment_runner.git

Note that experiment_runner installs mo-stable-baselines3 as a dependency.
"""
import argparse
from typing import Dict, Any

import numpy as np
from experiment_runner import ExperimentManager
from stable_baselines3.common.buffers import MoRolloutBuffer
from stable_baselines3.lppo import LPPO
from envs import *
from envs import GymDoors, GymSokoban, GymUnbreakableBottles, GymBreakableBottles


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value  # Linearly decrease

    return func

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="breakable")
args = parser.parse_args()
env_chosen = args.env

str2cls = {
    "unbreakable": GymUnbreakableBottles,
    "breakable": GymBreakableBottles,
    "sokoban": GymSokoban,
    "doors": GymDoors,
}
env_cls = str2cls[env_chosen]

class SASafetyHPT(ExperimentManager):

    def build_model(self, env, model_args: Dict[str, Any]) -> Any:
        model_args["learning_rate"] = linear_schedule(model_args["learning_rate"])
        model_args["batch_size"] = model_args["n_steps"] * 5
        model_args["n_epochs"] = 20
        model_args["policy_kwargs"] = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            n_objectives=2,
        )
        model_args["rollout_buffer_class"] = MoRolloutBuffer
        model_args["eta_values"] = 5.0
        model_args["beta_values"] = [2.0, 1.0]
        model_args["verbose"] = 0
        model_args["clip_range_vf"] = 0.2

        model = LPPO("MoMlpPolicy", env, 2, device="cpu", **model_args)
        return model

    def build_env(self, args: Dict[str, Any]) -> Any:
        return env_cls(mode="vector", **args)

    def evaluate(self, model, env, exp_args: Dict[str, Any]) -> float:
        scalarised_reward = 0
        for ep in range(exp_args["n_eval_episodes"]):
            obs, _ = env.reset()
            for _ in range(50):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action.item())
                scalarised_reward += np.dot(reward, [10, 1])
                if done:
                    break
        return np.round(scalarised_reward/exp_args["n_eval_episodes"], 3).astype(float)

if __name__ == "__main__":
    em = SASafetyHPT("first_try", f"{env_chosen}HPT400k", hp_path="hp_lexico.yaml", tb_log=True, normalize_reward=True, n_objectives=2)
    em.optimize(10)