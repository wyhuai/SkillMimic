import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import re
from skill.SkillMimiclab.skillmimic.utils import torch_utils

class MotionDataHandler:
    def __init__(self, motion_file, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False):
        self.device = device
        self._key_body_ids = key_body_ids
        self.cfg = cfg
        self.init_vel = init_vel
        self.play_dataset = play_dataset #V1
        self.max_episode_length = max_episode_length
        
        self.hoi_data_dict = {}
        self.hoi_data_label_batch = None
        self.motion_lengths = None
        self.load_motion(motion_file)

        self.num_envs = num_envs
        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.reward_weights_default = reward_weights_default
        self.reward_weights = {}
        self.reward_weights["p"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["p"])
        self.reward_weights["r"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["r"])
        self.reward_weights["op"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["op"])
        self.reward_weights["ig"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["ig"])
        self.reward_weights["cg1"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg1"])
        self.reward_weights["cg2"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg2"])
        self.reward_weights["pv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["pv"])
        self.reward_weights["rv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["rv"])
        self.reward_weights["or"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["or"])
        self.reward_weights["opv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["opv"])
        self.reward_weights["orv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["orv"])

    def load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1]
        all_seqs = glob.glob(motion_file + '/*.pt')
        self.num_motions = len(all_seqs)
        self.motion_lengths = torch.zeros(len(all_seqs), device=self.device, dtype=torch.long)
        motion_class = np.zeros(len(all_seqs), dtype=int)
        self.layup_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)
        self.root_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)

        all_seqs.sort(key=self._sort_key)
        for i, seq_path in enumerate(all_seqs):
            loaded_dict = self._process_sequence(seq_path)
            self.hoi_data_dict[i] = loaded_dict
            self.motion_lengths[i] = loaded_dict['hoi_data'].shape[0]
            motion_class[i] = int(loaded_dict['hoi_data_text'])
            if self.skill_name in ['layup', "SHOT_up"]:
                layup_target_ind = torch.argmax(loaded_dict['obj_pos'][:, 2])
                self.layup_target[i] = loaded_dict['obj_pos'][layup_target_ind]
                self.root_target[i] = loaded_dict['root_pos'][layup_target_ind]
        self._compute_motion_weights(motion_class)
        if self.play_dataset:
            self.max_episode_length = self.motion_lengths.min() - 1
    
    def _sort_key(self, filename):
        match = re.search(r'\d+.pt$', filename)
        return int(match.group().replace('.pt', '')) if match else -1

    def _process_sequence(self, seq_path):
        loaded_dict = {}
        hoi_data = torch.load(seq_path)
        loaded_dict['hoi_data_text'] = os.path.basename(seq_path)[0:3]
        loaded_dict['hoi_data'] = hoi_data.detach().to(self.device)
        data_frames_scale = self.cfg["env"]["dataFramesScale"]
        fps_data = self.cfg["env"]["dataFPS"] * data_frames_scale

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = self._compute_velocity(loaded_dict['root_pos'], fps_data)

        loaded_dict['root_rot_3d'] = loaded_dict['hoi_data'][:, 3:6].clone()
        loaded_dict['root_rot'] = torch_utils.exp_map_to_quat(loaded_dict['root_rot_3d']).clone()
        self.smooth_quat_seq(loaded_dict['root_rot'])

        q_diff = torch_utils.quat_multiply(
            torch_utils.quat_conjugate(loaded_dict['root_rot'][:-1, :].clone()), 
            loaded_dict['root_rot'][1:, :].clone()
        )
        angle, axis = torch_utils.quat_to_angle_axis(q_diff)
        exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
        loaded_dict['root_rot_vel'] = self._compute_velocity(exp_map, fps_data)

        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+156].clone()
        loaded_dict['dof_pos_vel'] = self._compute_velocity(loaded_dict['dof_pos'], fps_data)

        data_length = loaded_dict['hoi_data'].shape[0]
        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 165: 165+53*3].clone().view(data_length, 53, 3)
        loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids, :].view(data_length, -1).clone()
        loaded_dict['key_body_pos_vel'] = self._compute_velocity(loaded_dict['key_body_pos'], fps_data)

        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318+6:321+6].clone()
        loaded_dict['obj_pos_vel'] = self._compute_velocity(loaded_dict['obj_pos'], fps_data)

        loaded_dict['obj_rot'] = -loaded_dict['hoi_data'][:, 321+6:324+6].clone()
        loaded_dict['obj_rot_vel'] = self._compute_velocity(loaded_dict['obj_rot'], fps_data)
        if self.init_vel:
            loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
        loaded_dict['obj_rot'] = torch_utils.exp_map_to_quat(-loaded_dict['hoi_data'][:, 327:330]).clone()

        loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 330+6:331+6].clone())

        loaded_dict['hoi_data'] = torch.cat((
            loaded_dict['root_pos'],
            loaded_dict['root_rot_3d'],
            loaded_dict['dof_pos'],
            loaded_dict['dof_pos_vel'],
            loaded_dict['obj_pos'],
            loaded_dict['obj_rot'],
            loaded_dict['obj_pos_vel'],
            loaded_dict['key_body_pos'],
            loaded_dict['contact']
        ), dim=-1)
        
        return loaded_dict

    def _compute_velocity(self, positions, fps):
        velocity = (positions[1:, :].clone() - positions[:-1, :].clone()) * fps
        velocity = torch.cat((torch.zeros((1, positions.shape[-1])).to(self.device), velocity), dim=0)
        return velocity

    def smooth_quat_seq(self, quat_seq):
        n = quat_seq.size(0)

        for i in range(1, n):
            dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
            if dot_product < 0:
                quat_seq[i] *=-1

        return quat_seq

    def _compute_motion_weights(self, motion_class):
        unique_classes, counts = np.unique(motion_class, return_counts=True)
        class_to_index = {k: v for v, k in enumerate(unique_classes)}
        class_weights = 1 / counts
        indexed_classes = np.array([class_to_index[int(cls)] for cls in motion_class], dtype=int)
        self._motion_weights = class_weights[indexed_classes]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(torch.tensor(self._motion_weights), num_samples=n, replacement=True)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        lengths = self.motion_lengths[motion_ids].cpu().numpy()

        start = 2
        end = lengths - 2

        assert np.all(end > start) # Maybe some motions are too short to sample time properly.

        motion_times = np.random.randint(start, end + 1)  # +1  Because the upper limit of np.random.randint is an open interval

        motion_times = torch.tensor(motion_times, device=self.device, dtype=torch.int)

        if truncate_time is not None:
            assert truncate_time >= 0
            motion_times = torch.min(motion_times, self.motion_lengths[motion_ids] - truncate_time)

        if self.play_dataset:
            motion_times = torch.ones((1), device=self.device, dtype=torch.int32)
        return motion_times


    def get_initial_state(self, env_ids, motion_ids, start_frames):
        """
        Get the initial state for given motion_ids and start_frames.
        
        Parameters:
        motion_ids (Tensor): A tensor containing the motion id for each environment.
        start_frames (Tensor): A tensor containing the starting frame number for each environment.
        
        Returns:
        Tuple: A tuple containing the initial state
        """
        assert len(motion_ids) == len(env_ids)
        valid_lengths = self.motion_lengths[motion_ids] - start_frames if not self.play_dataset else self.motion_lengths[motion_ids]
        self.envid2episode_lengths[env_ids] = torch.where(valid_lengths < self.max_episode_length,
                                    valid_lengths, self.max_episode_length)

        # reward_weights_list = []
        hoi_data_list = []
        root_pos_list = []
        root_rot_list = []
        root_vel_list = []
        root_ang_vel_list = []
        dof_pos_list = []
        dof_vel_list = []
        obj_pos_list = []
        obj_pos_vel_list = []
        obj_rot_list = []
        obj_rot_vel_list = []

        for i, env_id in enumerate(env_ids):
            motion_id = motion_ids[i].item()
            start_frame = start_frames[i].item()

            self.envid2motid[env_id] = motion_id #V1
            episode_length = self.envid2episode_lengths[env_id].item()

            if self.hoi_data_dict[motion_id]['hoi_data_text'] == '000':
                state = self._get_special_case_initial_state(motion_id, start_frame, episode_length)
            else:
                state = self._get_general_case_initial_state(motion_id, start_frame, episode_length)

            # reward_weights_list.append(state['reward_weights'])
            for k in self.reward_weights_default:
                self.reward_weights[k][env_id] =  torch.tensor(state['reward_weights'][k], dtype=torch.float32, device=self.device)
            hoi_data_list.append(state["hoi_data"])
            root_pos_list.append(state['init_root_pos'])
            root_rot_list.append(state['init_root_rot'])
            root_vel_list.append(state['init_root_pos_vel'])
            root_ang_vel_list.append(state['init_root_rot_vel'])
            dof_pos_list.append(state['init_dof_pos'])
            dof_vel_list.append(state['init_dof_pos_vel'])
            obj_pos_list.append(state["init_obj_pos"])
            obj_pos_vel_list.append(state["init_obj_pos_vel"])
            obj_rot_list.append(state["init_obj_rot"])
            obj_rot_vel_list.append(state["init_obj_rot_vel"])

        hoi_data = torch.stack(hoi_data_list, dim=0)
        root_pos = torch.stack(root_pos_list, dim=0)
        root_rot = torch.stack(root_rot_list, dim=0)
        root_vel = torch.stack(root_vel_list, dim=0)
        root_ang_vel = torch.stack(root_ang_vel_list, dim=0)
        dof_pos = torch.stack(dof_pos_list, dim=0)
        dof_vel = torch.stack(dof_vel_list, dim=0)
        obj_pos = torch.stack(obj_pos_list, dim =0)
        obj_pos_vel = torch.stack(obj_pos_vel_list, dim =0)
        obj_rot = torch.stack(obj_rot_list, dim =0)
        obj_rot_vel = torch.stack(obj_rot_vel_list, dim =0)

        return hoi_data, \
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, \
                obj_pos, obj_pos_vel, obj_rot, obj_rot_vel
                

    def _get_special_case_initial_state(self, motion_id, start_frame, episode_length):
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)
        )

        return {
            "reward_weights": self._get_special_case_reward_weights(),
            "hoi_data": hoi_data,
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],
            "init_obj_pos": (torch.rand(3, device=self.device) * 10 - 5),
            "init_obj_pos_vel": torch.rand(3, device=self.device) * 5,
            "init_obj_rot": torch.rand(4, device=self.device),
            "init_obj_rot_vel": torch.rand(4, device=self.device) * 0.1
        }

    def _get_general_case_initial_state(self, motion_id, start_frame, episode_length):
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)
        )

        return {
            "reward_weights": self._get_general_case_reward_weights(),
            "hoi_data": hoi_data,
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],
            "init_obj_pos": self.hoi_data_dict[motion_id]['obj_pos'][start_frame, :],
            "init_obj_pos_vel": self.hoi_data_dict[motion_id]['obj_pos_vel'][start_frame, :],
            "init_obj_rot": self.hoi_data_dict[motion_id]['obj_rot'][start_frame, :],
            "init_obj_rot_vel": self.hoi_data_dict[motion_id]['obj_rot_vel'][start_frame, :]
        }

    def _get_special_case_reward_weights(self):
        reward_weights = self.reward_weights_default
        return {
            "p": reward_weights["p"],
            "r": reward_weights["r"],
            "op": reward_weights["op"] * 0.,
            "ig": reward_weights["ig"] * 0.,
            "cg1": reward_weights["cg1"] * 0.,
            "cg2": reward_weights["cg2"] * 0.,
            "pv": reward_weights["pv"],
            "rv": reward_weights["rv"],
            "or": reward_weights["or"],
            "opv": reward_weights["opv"],
            "orv": reward_weights["orv"],
        }

    def _get_general_case_reward_weights(self):
        reward_weights = self.reward_weights_default
        return {
            "p": reward_weights["p"],
            "r": reward_weights["r"],
            "op": reward_weights["op"],
            "ig": reward_weights["ig"],
            "cg1": reward_weights["cg1"],
            "cg2": reward_weights["cg2"],
            "pv": reward_weights["pv"],
            "rv": reward_weights["rv"],
            "or": reward_weights["or"],
            "opv": reward_weights["opv"],
            "orv": reward_weights["orv"],
        }

