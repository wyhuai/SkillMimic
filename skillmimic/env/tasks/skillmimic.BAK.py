from enum import Enum
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject


class SkillMimicBallPlay(HumanoidWholeBodyWithObject): 
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = SkillMimicBallPlay.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        # self._reset_default_env_ids = []
        # self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.robot_type = cfg["env"]["asset"]["assetFileName"]
        self.reward_weights_default = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.ref_hoi_obs_size = 323 + len(self.cfg["env"]["keyBodies"])*3 + 6 #V1
        
        self._load_motion(self.motion_file) #ZC1
        
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self.reward_weights = {}
        self.reward_weights["p"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["p"])
        self.reward_weights["r"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["r"])
        self.reward_weights["op"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["op"])
        self.reward_weights["ig"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["ig"])
        self.reward_weights["cg1"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg1"])
        self.reward_weights["cg2"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg2"])

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
        
        self.condition_size = 64
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)

        self._subscribe_events_for_change_condition()

        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) #{}
        self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.show_motion_test = False
        # self.init_from_frame_test = 0 #2 #ZC3
        self.motion_id_test = 0
        # self.options = [i for i in range(6) if i != 2]
        self.succ_pos = []
        self.fail_pos = []
        self.reached_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) #metric torch.bool

        self.show_abnorm = [0] * self.num_envs #V1

        return

    def post_physics_step(self):
        self._update_condition()

        super().post_physics_step()

        # self._compute_hoi_observations()
        self._update_hist_hoi_obs()

        return

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        
        obs_size += self.condition_size
        return obs_size

    def get_task_obs_size(self):
        return 0
    
    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): #Z
            textemb_batch = self.hoi_data_label_batch
            obs = torch.cat((obs,textemb_batch),dim=-1)
            self.obs_buf[:] = obs

            env_ids = torch.arange(self.num_envs)
            ts = self.progress_buf.clone() #self.progress_buf[0].clone()
            self._curr_ref_obs = self.hoi_data_batch[env_ids,ts].clone() #ZC0

        else:
            textemb_batch = self.hoi_data_label_batch[env_ids]
            obs = torch.cat((obs,textemb_batch),dim=-1)
            self.obs_buf[env_ids] = obs

            ts = self.progress_buf[env_ids].clone() #self.progress_buf[env_ids][0].clone()
            self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone() #ZC0

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self.envid2episode_lengths
                                                   )
        return
    
    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._hist_obs,
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  len(self._key_body_ids),
                                                  self.reward_weights
                                                  )
        return
    
    def smooth_quat_seq(self, quat_seq):
        n = quat_seq.size(0)

        for i in range(1, n):
            dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
            if dot_product < 0:
                quat_seq[i] *=-1

        return quat_seq
    

    def _load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1] #metric

        # '''load HOI dataset'''
        self.hoi_data_dict = {}

        grab_path = motion_file
        all_seqs = glob.glob(grab_path + '/*.pt')
        self.my_data_dict = {}

        data_idx = 0
        self.motion_lengths = torch.zeros(len(all_seqs), device=self.device, dtype=torch.long) #Z {}
        # self.motion_class = torch.zeros(len(all_seqs), device=self.device, dtype=torch.int)
        motion_class = np.zeros(len(all_seqs), dtype=int)
        self.layup_target = torch.zeros((len(all_seqs),3), device=self.device, dtype=torch.float) #metric
        self.root_target = torch.zeros((len(all_seqs),3), device=self.device, dtype=torch.float) 

        import re
        def sort_key(filename):
            match = re.search(r'\d+.pt$', filename)
            if match:
                return int(match.group().replace('.pt', ''))
            else:
                return -1
        all_seqs.sort(key = sort_key)

        self.max_episode_length = 600 if self.skill_name in ['run'] else 400 #100 + 400 #ZC30
        for i in range(len(all_seqs)):
            loaded_dict = {}
            hoi_data = torch.load(all_seqs[i])
            loaded_dict['hoi_data_text'] = os.path.basename(all_seqs[i])[0:3] #ZC 29:29+3
            print("load data:",loaded_dict['hoi_data_text'],hoi_data.shape[0],all_seqs[i])
            loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

            # loaded_dict['hoi_data'] = torch.cat(
            #     (
            #         loaded_dict['hoi_data'],
            #         loaded_dict['hoi_data'][-1:].repeat(40 + 200,1)
            #     ), dim=0
            # )
            
            # if loaded_dict['hoi_data_text'] == '1':
            #     loaded_dict['hoi_data'] = torch.cat(
            #         (
            #             loaded_dict['hoi_data'],
            #             loaded_dict['hoi_data'][-1:].repeat(20,1)
            #         ), dim=0
            #     )

            # '''change the data framerate'''
            # NOTE: this is used for temporary testing, and is not rigorous that may yield incorrect rotations.
            dataFramesScale = self.cfg["env"]["dataFramesScale"]
            # scale_hoi_data = torch.nn.functional.interpolate(loaded_dict['hoi_data'].unsqueeze(1).transpose(0,2), scale_factor=dataFramesScale, mode='linear', align_corners=True)
            # loaded_dict['hoi_data'] = scale_hoi_data.transpose(0,2).squeeze(1).clone().contiguous()

            if self.play_dataset==True:
                self.max_episode_length = loaded_dict['hoi_data'].shape[0]
            self.motion_played_length = loaded_dict['hoi_data'].shape[0] #fid

            self.fps_data = self.cfg["env"]["dataFPS"]*dataFramesScale

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'),loaded_dict['root_pos_vel']),dim=0)

            ############################
            loaded_dict['root_rot_3d'] = loaded_dict['hoi_data'][:, 3:6].clone()
            # print(loaded_dict['root_rot_3d']) #ZC5

            loaded_dict['root_rot'] = torch_utils.exp_map_to_quat(loaded_dict['root_rot_3d']).clone()
            self.smooth_quat_seq(loaded_dict['root_rot'])
            
            q_diff = torch_utils.quat_multiply(torch_utils.quat_conjugate(loaded_dict['root_rot'][:-1,:].clone()), loaded_dict['root_rot'][1:,:].clone())
            angle, axis = torch_utils.quat_to_angle_axis(q_diff)
            exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
            loaded_dict['root_rot_vel'] = exp_map*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'),loaded_dict['root_rot_vel']),dim=0)



            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+156].clone()
            loaded_dict['dof_pos_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_pos_vel'].shape[-1])).to('cuda'),loaded_dict['dof_pos_vel']),dim=0)

            data_length = loaded_dict['hoi_data'].shape[0]
            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 165: 165+53*3].clone().view(data_length,53,3)
            loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids, :].view(data_length,-1).clone()
            loaded_dict['key_body_pos_vel'] = (loaded_dict['key_body_pos'][1:,:].clone() - loaded_dict['key_body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['key_body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['key_body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['key_body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318+6:321+6].clone()
            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to('cuda'),loaded_dict['obj_pos_vel']),dim=0) 

            loaded_dict['obj_rot'] = -loaded_dict['hoi_data'][:, 321+6:324+6].clone()
            loaded_dict['obj_rot_vel'] = (loaded_dict['obj_rot'][1:,:].clone() - loaded_dict['obj_rot'][:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to('cuda'),loaded_dict['obj_rot_vel']),dim=0)
            loaded_dict['obj_rot'] = torch_utils.exp_map_to_quat(-loaded_dict['hoi_data'][:, 321+6:324+6]).clone()

            loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 330+6:331+6].clone())
            # loaded_dict['contact'] = torch.ones(loaded_dict['hoi_data'].shape[0],1).to('cuda')
            assert(torch.any(torch.abs(loaded_dict['contact'][:,0] - 0.5) <= 0.5, dim=0))
            

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(),
                                                    loaded_dict['root_rot_3d'].clone(),
                                                    loaded_dict['dof_pos'].clone(),
                                                    loaded_dict['dof_pos_vel'].clone(),
                                                    loaded_dict['obj_pos'].clone(),
                                                    loaded_dict['obj_rot'].clone(),
                                                    loaded_dict['obj_pos_vel'].clone(),
                                                    loaded_dict['key_body_pos'][:,:].clone(),
                                                    loaded_dict['contact'].clone()
                                                    ),dim=-1)

            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])

            self.hoi_data_dict[i] = loaded_dict
            data_idx += 1 
            self.motion_lengths[i] = loaded_dict['hoi_data'].shape[0] #Z
            motion_class[i] = int(loaded_dict['hoi_data_text'])
            if self.skill_name in ['layup', "SHOT_up"]: #metric
                layup_target_ind = torch.argmax(loaded_dict['obj_pos'][:,2])
                self.layup_target[i] = loaded_dict['obj_pos'][layup_target_ind]
                self.root_target[i] = loaded_dict['root_pos'][layup_target_ind]


        self.num_motions = data_idx

        unique_classes, counts = np.unique(motion_class, return_counts=True)
        class_to_index = {k: v for v, k in enumerate(unique_classes)}
        class_weights = 1 / counts
        # if 1 in class_to_index:
        #     class_weights[class_to_index[1]] *= 2
        indexed_classes = np.array([class_to_index[int(cls)] for cls in motion_class], dtype=int)
        self.motion_weights = class_weights[indexed_classes]

        return
    


    def _subscribe_events_for_change_condition(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "pick")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "layup")
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "rrun")
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "run")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "getup")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "shot_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "shot_up")
        
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "RunL")
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "RunR")
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "Run")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "011")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "012")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "013")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "001")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "002")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "031")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "032")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "033")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "034")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "035")
        
        return
    
    
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def _reset_envs(self, env_ids):
        # self._reset_default_env_ids = []
        # self._reset_ref_env_ids = []

        if(len(env_ids)>0): #metric
            self.reached_target[env_ids] = 0
        
        super()._reset_envs(env_ids)

        return

    def _reset_humanoid(self, env_ids):
        if self._state_init == SkillMimicBallPlay.StateInit.Start \
              or self._state_init == SkillMimicBallPlay.StateInit.Random:
            self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        return


    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        if (self._state_init == SkillMimicBallPlay.StateInit.Random
            or self._state_init == SkillMimicBallPlay.StateInit.Hybrid):
            motion_times = torch.randint(0, self.hoi_data_dict[0]['hoi_data'].shape[0]-2, (num_envs,), device=self.device, dtype=torch.long)
        elif (self._state_init == SkillMimicBallPlay.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)#.int()

        self.motion_times = motion_times.clone()

        # self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) #{}
        # self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # self.envid2idt = {}


        #TODO: i should has shape of env_ids
        for i in env_ids: #range(self.num_envs):

            # id_motion = self.motion_id_test #fid
            # id_motion = int(os.environ.get('clip', '-1')) 
            # id_motion = random.randint(0, self.num_motions-1)
            id_motion = np.random.choice(np.arange(self.num_motions), p=self.motion_weights/self.motion_weights.sum())
            # id_motion = int(i % self.num_motions)
            # id_motion = random.choice(self.options)
            self.envid2motid[i] = id_motion
            
            # id_t = self.init_from_frame_test #ZC3
            # id_t = self.motion_lengths[id_motion] - 6 + random.randint(-10, 4)
            # id_t = random.randint(2,  - 2) ###??wrongly added??
            # id_t = random.randint(2, self.motion_lengths[id_motion] - 2)
            #id_t = random.randint(2, 20)
            id_t = 2
            # id_t = random.randint(2, self.motion_lengths[id_motion] - 2) \
            #     if self.skill_name not in ['layup', 'SHOT_up'] else random.randint(3, 10)
            # id_t = 2
            # if random.randint(0,1) == 0:
            #     id_t = random.randint(2, self.motion_lengths[id_motion] - 2)
            # else:
            #     id_t = 2  #+ random.randint(0, 4)

            if id_t + self.max_episode_length > self.motion_lengths[id_motion]:
                # id_t = 0
                self.envid2episode_lengths[i] = self.motion_lengths[id_motion] - id_t
            else:
                # id_t = random.randint(1, self.motion_lengths[id_motion] - self.max_episode_length)
                self.envid2episode_lengths[i] = self.max_episode_length
            # id_t = random.randint(1, self.hoi_data_dict[id_motion]['hoi_data'].shape[0]-self.max_episode_length) 
            # self.envid2idt[i] = id_t


            #for class control
            # print(self.hoi_data_dict[id_motion]['hoi_data_text']) #Z20
            self.hoi_data_label_batch[i] = torch.nn.functional.one_hot(torch.tensor(int(self.hoi_data_dict[id_motion]['hoi_data_text'])).to("cuda"), num_classes=self.condition_size)#id_motion 
            # self.hoi_data_label_batch[i] = torch.nn.functional.one_hot(torch.tensor(61).to("cuda"), num_classes=self.condition_size)#id_motion 

            #for amplitude control
            # self.hoi_data_label_batch[i] = torch.tensor(int(self.hoi_data_dict[id_motion]['hoi_data_text'])).to("cuda").repeat(self.condition_size)
            # self.hoi_data_label_batch[i] = torch.tensor(1).to("cuda").repeat(self.condition_size) 

            #for robust standup
            if self.hoi_data_dict[id_motion]['hoi_data_text'] == '000': #Z

                # disable object rewards
                self.reward_weights["p"][i] = self.reward_weights_default["p"]
                self.reward_weights["r"][i] = self.reward_weights_default["r"]
                self.reward_weights["op"][i] = self.reward_weights_default["op"]*0.
                self.reward_weights["ig"][i] = self.reward_weights_default["ig"]*0.
                self.reward_weights["cg1"][i] = self.reward_weights_default["cg1"]*0.
                self.reward_weights["cg2"][i] = self.reward_weights_default["cg2"]*0.

                # self.hoi_data_batch[i] = self.hoi_data_dict[id_motion]['hoi_data'][id_t:id_t+self.max_episode_length]
                self.hoi_data_batch[i] = torch.nn.functional.pad( #ZC
                    self.hoi_data_dict[id_motion]['hoi_data'][id_t: id_t+self.envid2episode_lengths[i]],
                    (0, 0, 0, self.max_episode_length - self.envid2episode_lengths[i])
                    )

                self.init_root_pos[i] = self.hoi_data_dict[id_motion]['root_pos'][id_t,:]

                self.init_root_rot[i] = self.hoi_data_dict[id_motion]['root_rot'][id_t,:]
                self.init_dof_pos[i] = self.hoi_data_dict[id_motion]['dof_pos'][id_t,:] #+ (torch.rand_like(self.hoi_data_dict[id_motion]['dof_pos'][id_t,:])-0.5)*1
                self.init_root_pos_vel[i] = self.hoi_data_dict[id_motion]['root_pos_vel'][id_t,:]
                self.init_root_rot_vel[i] = self.hoi_data_dict[id_motion]['root_rot_vel'][id_t,:]
                self.init_dof_pos_vel[i] = self.hoi_data_dict[id_motion]['dof_pos_vel'][id_t,:]

                self.init_obj_pos[i][0] = (torch.rand_like(self.init_obj_pos[i][0])-0.5)*10
                self.init_obj_pos[i][1] = (torch.rand_like(self.init_obj_pos[i][0])-0.5)*10
                self.init_obj_pos[i][2] = torch.rand_like(self.init_obj_pos[i][0])*5
                self.init_obj_pos_vel[i] = torch.rand_like(self.init_obj_pos_vel[i])*5
                self.init_obj_rot[i] = torch.rand_like(self.init_obj_rot[i])
                self.init_obj_rot_vel[i] = torch.rand_like(self.init_obj_rot_vel[i])*0.1

            # for general skill learning
            else:
                # enable full rewards
                self.reward_weights["p"][i] = self.reward_weights_default["p"]
                self.reward_weights["r"][i] = self.reward_weights_default["r"]
                self.reward_weights["op"][i] = self.reward_weights_default["op"]
                self.reward_weights["ig"][i] = self.reward_weights_default["ig"]
                self.reward_weights["cg1"][i] = self.reward_weights_default["cg1"]
                self.reward_weights["cg2"][i] = self.reward_weights_default["cg2"]

                self.hoi_data_batch[i] = torch.nn.functional.pad( #ZC
                    self.hoi_data_dict[id_motion]['hoi_data'][id_t: id_t+self.envid2episode_lengths[i]],
                    (0, 0, 0, self.max_episode_length - self.envid2episode_lengths[i])
                    ) #Z? :id_t+self.max_episode_length

                self.init_root_pos[i] = self.hoi_data_dict[id_motion]['root_pos'][id_t,:]

                self.init_root_rot[i] = self.hoi_data_dict[id_motion]['root_rot'][id_t,:]
                self.init_dof_pos[i] = self.hoi_data_dict[id_motion]['dof_pos'][id_t,:]
                self.init_root_pos_vel[i] = self.hoi_data_dict[id_motion]['root_pos_vel'][id_t,:]
                self.init_root_rot_vel[i] = self.hoi_data_dict[id_motion]['root_rot_vel'][id_t,:]

                self.init_dof_pos_vel[i] = self.hoi_data_dict[id_motion]['dof_pos_vel'][id_t,:]

                self.init_obj_pos[i] = self.hoi_data_dict[id_motion]['obj_pos'][id_t,:]
                self.init_obj_pos_vel[i] = self.hoi_data_dict[id_motion]['obj_pos_vel'][id_t,:]
                self.init_obj_rot[i] = self.hoi_data_dict[id_motion]['obj_rot'][id_t,:]
                self.init_obj_rot_vel[i] = self.hoi_data_dict[id_motion]['obj_rot_vel'][id_t,:]


        if self.show_motion_test == False:
            print('motionid:', self.hoi_data_dict[int(self.envid2motid[0])]['hoi_data_text'], \
                'motionlength:', self.hoi_data_dict[int(self.envid2motid[0])]['hoi_data'].shape[0]) #ZC
            self.show_motion_test = True


        self._set_env_state(env_ids=env_ids, 
                    root_pos=self.init_root_pos,
                    root_rot=self.init_root_rot,
                    dof_pos=self.init_dof_pos,
                    root_vel=self.init_root_pos_vel,
                    root_ang_vel=self.init_root_rot_vel,
                    dof_vel=self.init_dof_pos_vel,
                    )

        return
    
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = root_rot[env_ids]
        self._humanoid_root_states[env_ids, 7:10] = root_vel[env_ids]
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel[env_ids]
        
        self._dof_pos[env_ids] = dof_pos[env_ids]
        self._dof_vel[env_ids] = dof_vel[env_ids]
        return

    
    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                               self._hist_obs,
                                                               self.fps_data,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                   self._hist_obs[env_ids],
                                                                   self.fps_data,
                                                                   self.progress_buf[env_ids])
            
        # self._hist_obs = self._curr_obs.clone()
        
        return
    
    def _update_condition(self):
        for evt in self.envts:

            if evt.action == "pick" and evt.value > 0:
                # self.gym.set_sim_rigid_body_states(self.sim, self._proj_states, gymapi.STATE_ALL)
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(1).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(0.2).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size) 
                self.control_signal = 1

            elif (evt.action == "layup") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(2).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(1.4).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size) 
                self.control_signal = 7

            elif (evt.action == "rrun") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(3).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(1.0).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size) 

            elif (evt.action == "run") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(4).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(0.6).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size) 
            
            elif (evt.action == "getup") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(5).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(0.6).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size) 
                self.control_signal = 0

            elif (evt.action == "shot_down") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(3).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(1.0).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size)
                self.control_signal = 5

            elif (evt.action == "shot_up") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(4).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
                # self.hoi_data_label_batch = torch.tensor(0.6).to("cuda").repeat(self.hoi_data_label_batch.shape[0],self.condition_size)
                self.control_signal = 6
            
            
            elif (evt.action == "RunL") and evt.value > 0:
                self.control_signal = 2 #1
            elif (evt.action == "RunR") and evt.value > 0:
                self.control_signal = 3 #2
            elif (evt.action == "Run") and evt.value > 0:
                self.control_signal = 4 #3
            elif (evt.action == "2Run") and evt.value > 0:
                self.control_signal = 9 #1
            elif (evt.action == "SHOT") and evt.value > 0:
                self.control_signal = 8 #3

            elif (evt.action == "011") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(11).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "012") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(12).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "013") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(13).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "001") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(1).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "002") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(2).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "031") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(31).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "032") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(32).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "033") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(33).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "034") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(34).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
            elif (evt.action == "035") and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(35).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)

                
    
    def play_dataset_step(self, time): #Z12

        t = time

        for env_id, env_ptr in enumerate(self.envs):
            # t += self.envid2idt[env_id]

            ### update object ###
            motid = self.envid2motid[env_id].item()
            self._target_states[env_id, :3] = self.hoi_data_dict[motid]['obj_pos'][t,:]
            self._target_states[env_id, 3:7] = self.hoi_data_dict[motid]['obj_rot'][t,:]
            self._target_states[env_id, 7:10] = torch.zeros_like(self._target_states[env_id, 7:10])
            self._target_states[env_id, 10:13] = torch.zeros_like(self._target_states[env_id, 10:13])

            ### update subject ###   
            _humanoid_root_pos = self.hoi_data_dict[motid]['root_pos'][t,:].clone()
            _humanoid_root_rot = self.hoi_data_dict[motid]['root_rot'][t,:].clone()
            self._humanoid_root_states[env_id, 0:3] = _humanoid_root_pos
            self._humanoid_root_states[env_id, 3:7] = _humanoid_root_rot
            self._humanoid_root_states[env_id, 7:10] = torch.zeros_like(self._humanoid_root_states[env_id, 7:10])
            self._humanoid_root_states[env_id, 10:13] = torch.zeros_like(self._humanoid_root_states[env_id, 10:13])
            
            self._dof_pos[env_id] = self.hoi_data_dict[motid]['dof_pos'][t,:].clone()
            # self._dof_pos[:,108:156] = 0
            self._dof_vel[env_id] = torch.zeros_like(self._dof_vel[env_id])

            # env_id_int32 = self._humanoid_actor_ids[env_id].unsqueeze(0)



            contact = self.hoi_data_dict[motid]['contact'][t,:]
            obj_contact = torch.any(contact > 0.1, dim=-1)
            root_rot_vel = self.hoi_data_dict[motid]['root_rot_vel'][t,:]
            # angle, _ = torch_utils.exp_map_to_angle_axis(root_rot_vel)
            angle = torch.norm(root_rot_vel)
            abnormal = torch.any(torch.abs(angle) > 5.) #Z

            if abnormal == True:
                print("frame:", t, "abnormal:", abnormal, "angle", angle)
                # print(" ", self.hoi_data_dict[motid]['root_rot_vel'][t])
                # print(" ", angle)
                self.show_abnorm[env_id] = 10

            handle = self._target_handles[env_id]
            if obj_contact == True:
                # print(t, "contact")
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1., 0., 0.))
            else:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0., 1., 0.))
            
            if abnormal == True or self.show_abnorm[env_id] > 0: #Z
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.)) 
                self.show_abnorm[env_id] -= 1
            else:
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.)) 
      
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._refresh_sim_tensors()     

        self.render(t=time)
        self.gym.simulate(self.sim)

        self._compute_observations()

        # if t == 0: 
        #     self.keybodies = self._rigid_body_pos[:1, :, :]
            
        # else:
        #     self.keybodies = torch.cat((self.keybodies, self._rigid_body_pos[:1, :, :]),dim=0)      


        # if t>=(self.max_episode_length-1):
        #     hoi_data = torch.cat((
        #                                             self.hoi_data_dict[0]['root_pos'][:-1].clone(),
        #                                             self.hoi_data_dict[0]['root_rot_3d'][:-1].clone(),

        #                                             self.hoi_data_dict[0]['root_rot_3d'][:-1].clone(),
        #                                             self.hoi_data_dict[0]['dof_pos'][:-1].clone(),

        #                                             self.keybodies[1:].reshape(-1,53*3),

        #                                             self.hoi_data_dict[0]['obj_pos'][:-1].clone(),
        #                                             torch.zeros_like(self.hoi_data_dict[0]['obj_pos'][:-1]),
        #                                             torch.zeros_like(self.hoi_data_dict[0]['obj_pos'][:-1]),
        #                                             torch.zeros_like(self.hoi_data_dict[0]['obj_pos'][:-1]),
                                                    
        #                                             self.hoi_data_dict[0]['contact'][:-1].clone()
        #                                             ),dim=-1)

        #     save_hoi_data = hoi_data.clone()
        #     torch.save(save_hoi_data, 'skillmimic/data/motions/mocap_0330_labeled/'+'015'+'.pt') #ZC7 #projectname
        #     import sys
        #     sys.exit(0)
        
        # if t>=(self.max_episode_length-1): #ZC9
        #     print(self.hoi_data_dict[0]['root_rot_3d']) #Z
        #     import sys
        #     sys.exit(0)

        return self.obs_buf
    
    def _draw_task_play(self,t):
        
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32) # color

        self.gym.clear_lines(self.viewer)

        starts = self.hoi_data_dict[0]['hoi_data'][t, :3]

        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self.hoi_data_dict[0]['key_body_pos'][t, j*3:j*3+3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)

        return

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
            
            # if t%4==0:
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]
                dataname = self.motion_file[len('skillmimic/data/motions/mocap_0330/'):-7] #ZC8 #projectname
                dataname = "smooth_ft1"#"hlc_bc_SHOT"
                rgb_filename = "skillmimic/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("skillmimic/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
        return
    
    def _draw_task(self):

        # # draw obj contact
        # obj_contact = torch.any(torch.abs(self._tar_contact_forces[..., 0:2]) > 0.1, dim=-1)
        # for env_id, env_ptr in enumerate(self.envs):
        #     env_ptr = self.envs[env_id]
        #     handle = self._target_handles[env_id]

        #     if obj_contact[env_id] == True:
        #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
        #                                     gymapi.Vec3(1., 0., 0.))
        #     else:
        #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
        #                                     gymapi.Vec3(0., 1., 0.))

        return




#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, target_states, hist_obs, fps, progress_buf):

    ## diffvel, set 0 for the first frame
    # hist_dof_pos = hist_obs[:,6:6+156]
    # dof_diffvel = (dof_pos - hist_dof_pos)*fps
    # dof_diffvel = dof_diffvel*(progress_buf!=1).to(float).unsqueeze(dim=-1)

    dof_vel = dof_vel*(progress_buf!=1).unsqueeze(dim=-1)

    contact = torch.zeros(key_body_pos.shape[0],1,device=dof_vel.device)
    obs = torch.cat((root_pos, torch_utils.quat_to_exp_map(root_rot), dof_pos, dof_vel, target_states[:,:10], key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), contact), dim=-1)
    return obs

# @torch.jit.script
def compute_humanoid_reward(hoi_ref, hoi_obs, hoi_obs_hist, contact_buf, tar_contact_forces, len_keypos, w): #ZCr
    ## type: (Tensor, Tensor, Tensor, Tensor, Int, float) -> Tensor

    ### data preprocess ###

    # simulated states
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:3+3]
    dof_pos = hoi_obs[:,6:6+52*3]
    dof_pos_vel = hoi_obs[:,162:162+52*3]
    obj_pos = hoi_obs[:,318:318+3]
    obj_rot = hoi_obs[:,321:321+4]
    obj_pos_vel = hoi_obs[:,325:325+3]
    key_pos = hoi_obs[:,328:328+len_keypos*3]
    contact = hoi_obs[:,-1:]# fake one
    key_pos = torch.cat((root_pos, key_pos),dim=-1)
    body_rot = torch.cat((root_rot, dof_pos),dim=-1)
    ig = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj_pos[:,:3]
    ig_wrist = ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ig = ig.transpose(0,1).view(-1,(len_keypos+1)*3)

    dof_pos_vel_hist = hoi_obs_hist[:,162:162+52*3] #ZC

    # reference states
    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:3+3]
    ref_dof_pos = hoi_ref[:,6:6+52*3]
    ref_dof_pos_vel = hoi_ref[:,162:162+52*3]
    ref_obj_pos = hoi_ref[:,318:318+3]
    ref_obj_rot = hoi_ref[:,321:321+4]
    ref_obj_pos_vel = hoi_ref[:,325:325+3]
    ref_key_pos = hoi_ref[:,328:328+len_keypos*3]
    ref_obj_contact = hoi_ref[:,-1:]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj_pos[:,:3]
    ref_ig_wrist = ref_ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos+1)*3)


    ### body reward ###

    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos)**2,dim=-1)
    # ep = torch.mean((ref_key_pos[:,0:(7+1)*3] - key_pos[:,0:(7+1)*3])**2,dim=-1) #ZC
    rp = torch.exp(-ep*w['p'])

    # body rot reward
    er = torch.mean((ref_body_rot - body_rot)**2,dim=-1)
    rr = torch.exp(-er*w['r'])

    # body pos vel reward
    epv = torch.zeros_like(ep)
    rpv = torch.exp(-epv*w['pv'])

    # body rot vel reward
    erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
    rrv = torch.exp(-erv*w['rv'])

    # body vel smoothness reward
    # e_vel_diff = torch.mean((dof_pos_vel - dof_pos_vel_hist)**2, dim=-1)
    # r_vel_diff = torch.exp(-e_vel_diff * 0.05) #w['vel_diff']
    e_vel_diff = torch.mean((dof_pos_vel - dof_pos_vel_hist)**2 / ((ref_dof_pos_vel + 1e-6)*1e6)**2, dim=-1)
    r_vel_diff = torch.exp(-e_vel_diff * 0.1) #w['vel_diff']


    rb = rp*rr*rpv*rrv *r_vel_diff #ZC
    # print(rp, rr, rpv, rrv) 


    ### object reward ###

    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    # object rot reward
    eor = torch.zeros_like(ep) #torch.mean((ref_obj_rot - obj_rot)**2,dim=-1)
    ror = torch.exp(-eor*w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])

    # object rot vel reward
    eorv = torch.zeros_like(ep) #torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv*w['orv'])

    ro = rop*ror*ropv*rorv


    ### interaction graph reward ###

    eig = torch.mean((ref_ig - ig)**2,dim=-1) #Zw
    # eig = torch.mean((ref_ig_wrist - ig_wrist)**2,dim=-1)
    rig = torch.exp(-eig*w['ig'])


    ### simplified contact graph reward ###

    # Since Isaac Gym does not yet provide API for detailed collision detection in GPU pipeline, 
    # we use force detection to approximate the contact status.
    # In this case we use the CG node istead of the CG edge for imitation.
    # TODO: update the code once collision detection API is available.

    ## body ids
    # Pelvis, 0 
    # L_Hip, 1 
    # L_Knee, 2
    # L_Ankle, 3
    # L_Toe, 4
    # R_Hip, 5 
    # R_Knee, 6
    # R_Ankle, 7
    # R_Toe, 8
    # Torso, 9
    # Spine, 10 
    # Spine1, 11
    # Chest, 12
    # Neck, 13
    # Head, 14
    # L_Thorax, 15
    # L_Shoulder, 16
    # L_Elbow, 17
    # L_Wrist, 18
    # L_Hand, 19-33
    # R_Thorax, 34 
    # R_Shoulder, 35
    # R_Elbow, 36
    # R_Wrist, 37
    # R_Hand, 38-52

    # body contact
    contact_body_ids = [0,1,2,5,6,9,10,11,12,13,14,15,16,17,34,35,36]
    body_contact_buf = contact_buf[:, contact_body_ids, :].clone()
    body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)
    body_contact = torch.all(body_contact, dim=-1).to(float) # =1 when no contact happens to the body

    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(float) # =1 when contact happens to the object

    ref_body_contact = torch.ones_like(ref_obj_contact) # no body contact for all time
    ecg1 = torch.abs(body_contact - ref_body_contact[:,0])
    rcg1 = torch.exp(-ecg1*w['cg1'])
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:,0])
    rcg2 = torch.exp(-ecg2*w['cg2'])

    rcg = rcg1*rcg2


    ### task-agnostic HOI imitation reward ###
    reward = rb*ro*rig*rcg
    # print(rb, ro, rig, rcg) #ZC

    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    # if (enable_early_termination):
    #     body_height = rigid_body_pos[:, 0, 2] # root height
    #     body_fall = body_height < termination_heights# [4096] 
    #     has_failed = body_fall.clone()
    #     has_failed *= (progress_buf > 1)
        
    #     terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    # reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated