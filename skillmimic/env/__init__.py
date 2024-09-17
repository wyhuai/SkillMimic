"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

from math import pi
import numpy as np
#from rialto.envs.isaac_env import IsaacGoalEnv
import gymnasium as gym #https://github.com/openai/gym

import gym
import os


gym.register(
    id="Skillmimic-v0",
    entry_point="skillmimic.env.tasks:BaseTask",
    kwargs={"cfg_entry_point": "skillmimic.data.cfg:GeneralEnvCfg"},
)

'''
def create_env(env_name, continuous_action_space=False,  max_path_length=50, display=False,render_images=False,img_shape=(64,64),usd_name="mugandbenchv2", randomize_object_name="", randomize_action_mag=None, usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/",  num_envs=1, sensors=["rgb"], num_cameras=1, euler_rot=True, randomize_rot=False,randomize_pos=False, cfg=None):
    """Helper function."""

    return IsaacGoalEnv(max_path_length=max_path_length, display=display,randomize_action_mag=randomize_action_mag, randomize_object_name=randomize_object_name, render_images=render_images,img_shape=img_shape, usd_name=usd_name, usd_path=usd_path, num_envs=num_envs, sensors=sensors,num_cameras=num_cameras, euler_rot=euler_rot, randomize_rot=randomize_rot, randomize_pos=randomize_pos, cfg=cfg)


def get_env_params(env_name, images=False):
    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_timesteps=1e6,
    )


    env_specific_params = dict(
        goal_threshold=0.05,
    )
    
    base_params.update(env_specific_params)
    return base_params
    '''