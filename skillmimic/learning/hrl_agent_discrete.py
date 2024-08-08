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

import copy
from datetime import datetime
import random
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

import learning.common_agent_discrete as common_agent_discrete
import learning.skillmimic_models as skillmimic_models #ZC0
import learning.skillmimic_network_builder as skillmimic_network_builder
import learning.skillmimic_agent as skillmimic_agent

from tensorboardX import SummaryWriter

class HRLAgentDiscrete(common_agent_discrete.CommonAgentDiscrete):
    def __init__(self, base_name, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']

        self._latent_dim = config['latent_dim'] #Z0 #V1
        self._control_mapping = config['control_mapping']
        
        self.delta_action = config['delta_action'] #ZC0

        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        self.resume_from = config['resume_from']

        return

    def train(self):
        if self.resume_from != 'None':
            self.restore(self.resume_from)
        super().train()
        
    def env_step(self, actions):
        # actions = self.preprocess_actions(actions)
        obs = self.obs['obs']

        rewards = 0.0
        # disc_rewards = 0.0 #ZC0
        done_count = 0.0
        terminate_count = 0.0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, actions)

            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)
            
            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']
            
            # amp_obs = infos['amp_obs']
            # curr_disc_reward = self._calc_disc_reward(amp_obs)
            # disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        # disc_rewards /= self._llc_steps

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        # infos['disc_rewards'] = disc_rewards

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        # self._disc_reward_w = config['disc_reward_w'] #ZC0
        self._imit_reward_w = config['imit_reward_w']

        return

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim #Z0 self._latent_dim + self.delta_action 
        return

    def init_tensors(self):
        super().init_tensors()

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['logits'] = torch.zeros(batch_shape + (self.actions_num,) \
                                                                   ,dtype= torch.long, device = self.experience_buffer.device)
        self.tensor_list += ['logits']

        del self.experience_buffer.tensor_dict['actions']
  
        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape, # + (1,), #ZC0
                                                                dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.tensor_list += ['disc_rewards']

        return

    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']

        # network_builder = ase_network_builder.ASEBuilder()
        network_builder = skillmimic_network_builder.SkillMimicBuilder() #ZC0
        network_builder.load(network_params)

        # network = ase_models.ModelASEContinuous(network_builder)
        network = skillmimic_models.SkillMimicModelContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        # self._llc_agent = ase_agent.ASEAgent('llc', llc_agent_config)
        self._llc_agent = skillmimic_agent.SkillMimicAgent('llc', llc_agent_config)

        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()

        self._llc_agent.model.to(self.device)
        
        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0] 
        obs_size -= self._task_size
        obs_size += 64 #Z
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[0], obs_space.high[0], shape=(obs_size,))

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer' : self.algo_observer}
        config['env_info'] = llc_env_info

        return config

    def _compute_llc_action(self, obs, actions):

        # print(actions)
        # controlmapping = torch.tensor([31,1,2,12,13,11]).to(self.device)#{0:31, 1:1, 2:2, 3:12, 4:13, 5:11}
        controlmapping = torch.tensor(self._control_mapping).to(self.device)
        actions = controlmapping[actions]

        llc_obs = self._extract_llc_obs(obs)
        # actions += 1 #ZC0
        control_signal = torch.zeros((llc_obs.size(0),64), device=llc_obs.device)
        control_signal[torch.arange(llc_obs.size(0)), -64 + (actions)] = 1.
        llc_obs = torch.cat((llc_obs, control_signal), dim=-1)

        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        # z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs) #, cls_latents=z
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs):
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward

    def _combine_rewards(self, task_rewards, disc_rewards): 
        combined_rewards = self._task_reward_w * task_rewards 
                        #  + self._disc_reward_w * disc_rewards #ZC0
                        # + self._imit_reward_w * 
        
        #combined_rewards = task_rewards * disc_rewards
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return