"""
In this script we will learn the reference policy for the given environment using the LPPO algorithm and the ExperimentRunner class from the libraries:
- https://github.com/maymac00/mo-stable-baselines3
- https://github.com/maymac00/experiment_runner
respectively.

Both can be installed using pip:
pip install git+https://github.com/maymac00/experiment_runner.git

Note that experiment_runner installs mo-stable-baselines3 as a dependency.
"""
from typing import Dict, Any

import numpy as np
from experiment_runner import ExperimentManager
from stable_baselines3.lppo import LPPO
from envs import *

class SASafetyHPT(ExperimentManager):

    def build_model(self, env, model_args: Dict[str, Any]) -> Any:
        model_args["batch_size"] = model_args["n_steps"] * 5
        model_args["n_epochs"] = 20
        model_args["policy_kwargs"] = dict(
            net_arch=dict(pi=[32, 32], vf=[32, 32]),

        )

        model = LPPO("MoMlpPolicy", env, device="cpu", **model_args)
        return model

    def build_env(self, args: Dict[str, Any]) -> Any:
        return GymBreakableBottles(mode="vector", **args)

    def evaluate(self, model, env, exp_args: Dict[str, Any]) -> float:
        scalarised_reward = 0
        for ep in range(exp_args["n_eval_episodes"]):
            obs, _ = env.reset()
            for _ in range(50):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                scalarised_reward += np.dot(reward, [10, 1])
                if done:
                    break
        return np.round(scalarised_reward/exp_args["n_eval_episodes"], 3).astype(float)

if __name__ == "__main__":
    em = SASafetyHPT("breakable", "metadriveHPT500k", hp_path="hp_lexico.yaml", tb_log=True, normalize_reward=True)
    em.optimize(10)