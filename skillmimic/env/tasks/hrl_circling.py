# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import Tensor
from typing import Tuple
from enum import Enum

from skill.SkillMimiclab.skillmimic.utils import torch_utils
from skill.SkillMimiclab.skillmimic.utils.motion_data_handler import MotionDataHandler

#from isaacgym import gymapi
#from isaacgym import gymtorch
#from isaacgym.torch_utils import *

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2

class HRLCircling(HumanoidWholeBodyWithObject):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = str(cfg["env"]["stateInit"])
        if state_init.lower() == "random":
            self._state_init = -1
            print("Random Reference State Init (RRSI)")
        else:
            self._state_init = int(state_init)
            print(f"Deterministic Reference State Init from {self._state_init}")

        self.goal_size = 5

        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.robot_type = cfg["env"]["asset"]["assetFileName"]
        self.reward_weights_default = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']

        # self.cfg["env"]["numActions"] = 66
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._load_motion(self.motion_file)

        self._goal_position = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._goal_radius = torch.zeros([self.num_envs, 1], device=self.device, dtype=torch.float)

        self._termination_heights = torch.tensor(self.cfg["env"]["terminationHeight"], device=self.device, dtype=torch.float)

        self.reached_target = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        
        return


    def get_task_obs_size(self):
        if (self._enable_task_obs):
            obs_size = self.goal_size
        else:
            obs_size = 0
        return obs_size

    def _load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1] #metric
        self.max_episode_length = 800
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]

        self._motion_data = MotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, self.max_episode_length, self.reward_weights_default, self.init_vel)

        return

    def _reset_actors(self, env_ids):
        if self._state_init == -1:
            self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        elif self._state_init >= 2:
            self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert(False), f"Unsupported state initialization from: {self._state_init}"

        super()._reset_actors(env_ids)

        return

    def _reset_humanoid(self, env_ids):
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
        self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]
        self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]
        
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        return

    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)

        _, \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        _, \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return
    
    def _reset_envs(self, env_ids):
        
        super()._reset_envs(env_ids)  

        if(len(env_ids)>0):
            n = len(env_ids)

            self._goal_position[env_ids, 0] = self._humanoid_root_states[env_ids, 0]
            self._goal_position[env_ids, 1] = self._humanoid_root_states[env_ids, 1]
            self._goal_radius[env_ids, :] = torch.rand(n,1).to("cuda")*3 + 2

            self.reached_target[env_ids] = False

        return


    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        obj_obs = self._compute_obj_obs(env_ids)

        if(self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([obs, obj_obs, task_obs], dim=-1)
        else:
            obs = torch.cat([obs, obj_obs], dim=-1)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return


    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_pos = self._humanoid_root_states[..., 0:3]
            goal_pos = self._goal_position
            goal_r = self._goal_radius
            root_rot = self._humanoid_root_states[..., 3:7]
        else:
            root_pos = self._humanoid_root_states[env_ids, 0:3]
            goal_pos = self._goal_position[env_ids]
            goal_r = self._goal_radius[env_ids]
            root_rot = self._humanoid_root_states[env_ids, 3:7]

        obs = compute_circling_observations(root_pos, goal_pos, root_rot, goal_r)

        return obs


    def _compute_reset(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces,self._rigid_body_pos,self._target_states[..., 0:3],
                                                                            root_pos, self._goal_position,
                                                                            self.max_episode_length, self._enable_early_termination, self._termination_heights
                                                                            )
        return
    
    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_vel = self._humanoid_root_states[..., 7:10]
        ball_pos = self._target_states[..., 0:3]
        ball_vel = self._target_states[..., 7:10]
        self.rew_buf[:] = compute_circling_reward(root_pos, root_vel, ball_pos, ball_vel, self._goal_position, self._goal_radius)

        # for test
        distance_to_goal = torch.norm(ball_pos[:, :2] - self._goal_position, dim=-1)
        at_target = (distance_to_goal < 0.3)
        self.reached_target = self.reached_target | at_target

        return
    
    def _draw_task(self):

        point_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Red for goal position

        self.gym.clear_lines(self.viewer)

        goal_positions = self._goal_position.cpu().numpy()
        goal_radius = self._goal_radius.cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            # Draw goal position as a small line segment (point)
            goal_pos = goal_positions[i]
            goal_r = goal_radius[i]
            goal_verts = np.array([goal_pos[0]-goal_r, goal_pos[1]-goal_r, 0.8, goal_pos[0] + goal_r, goal_pos[1] + goal_r, 0.8], dtype=np.float32)
            goal_verts = goal_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, goal_verts.shape[0], goal_verts, point_color)
            goal_verts = np.array([goal_pos[0]-goal_r, goal_pos[1]+goal_r, 0.8, goal_pos[0] + goal_r, goal_pos[1] - goal_r, 0.8], dtype=np.float32)
            goal_verts = goal_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, goal_verts.shape[0], goal_verts, point_color)

        return

    
    def _update_proj(self):
        # mouse control
        if self.projtype == 'Mouse':
            for evt in self.evts:
                if (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
                    x = torch.rand(self.num_envs).to("cuda")*6 + 2
                    y = torch.rand(self.num_envs).to("cuda")*6 + 2
                    self._goal_position[:, 0] = self._humanoid_root_states[:, 0]+x
                    self._goal_position[:, 1] = self._humanoid_root_states[:, 1]+y
                    self._goal_radius[:, :] = torch.rand(self.num_envs,1).to("cuda")*3 + 2                 
                    self.reached_target[:] = False
                print(evt.action)
        return
    
    def get_num_amp_obs(self):
        return 323 + len(self.cfg["env"]["keyBodies"])*3 + 6  #0
    
#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_circling_observations(root_pos, goal_pos, root_rot, goal_r):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot) #world: torch_utils.calc_heading_quat(root_rot)
    
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    local_facing_dir = torch_utils.quat_rotate(heading_rot, facing_dir)  

    local_tar_pos = goal_pos - root_pos[..., 0:2]
    local_tar_pos_3d = torch.cat([local_tar_pos, torch.zeros_like(local_tar_pos[..., 0:1])], dim=-1)  # Expands to 3D vectors
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos_3d)
    local_tar_pos = local_tar_pos[..., 0:2]
    
    # Calculate relative angle in radians
    angle = torch.atan2(local_tar_pos[:, 1], local_tar_pos[:, 0]) - torch.atan2(local_facing_dir[:, 1], local_facing_dir[:, 0])
    angle = torch_utils.normalize_angle(angle)
    
    # Compute cosine and sine of the angle
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    goal_angle = torch.stack([cos_angle, sin_angle], dim=-1)

    obs = torch.cat([local_tar_pos, goal_angle, goal_r], dim=-1) #world: goal_pos - root_pos[..., :2]

    return obs


# @torch.jit.script
def compute_circling_reward(root_pos, root_vel, ball_pos, ball_vel, goal_pos, goal_r):
    # # Body position rewards
    # distance_to_goal = torch.norm(root_pos[:, :2] - goal_pos, dim=-1)
    # d_error = torch.norm(distance_to_goal.unsqueeze(-1) - goal_r, dim=-1)
    # position_reward = torch.exp(-d_error)

    # Ball position rewards
    distance_to_goal = torch.norm(ball_pos[:, :2] - goal_pos, dim=-1)
    d_error = torch.norm(distance_to_goal.unsqueeze(-1) - goal_r, dim=-1)
    position_reward = torch.exp(-d_error)

    # still punish
    v = torch.norm(ball_vel,dim=-1)
    
    # When the velocity is less than 0.5, the reward is 0.1; when the speed is larger than or equal to 0.5, the reward is 1.0
    v_penalty = torch.where(v < 0.5, torch.tensor(0.1, device=root_pos.device), torch.tensor(1., device=root_pos.device)) # yry

    # Total reward
    reward = position_reward * v_penalty

    return reward


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos, ball_pos, root_pos, goal_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    
    terminated = torch.zeros_like(reset_buf)
    
    if (enable_early_termination):
        has_fallen = root_pos[..., 2] < termination_heights
        has_fallen *= (progress_buf > 1) # Same effect as using OR
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    # reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)

    return reset, terminated