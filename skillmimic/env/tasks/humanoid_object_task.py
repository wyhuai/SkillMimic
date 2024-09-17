from enum import Enum
import numpy as np
import torch
from torch import Tensor
import glob, os, random

#from isaacgym import gymtorch
#from isaacgym import gymapi
#from isaacgym.torch_utils import *
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg

from skill.SkillMimiclab.skillmimic.utils import torch_utils

from env.tasks.humanoid_task import HumanoidWholeBody
from skill.SkillMimiclab.skillmimic.utils.metrics import Metrics, compute_evaluation_metrics


PERTURB_PROJECTORS = [
    ["small", 60],
    # ["large", 60],
]

class HumanoidWholeBodyWithObject(HumanoidWholeBody): #metric
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.projtype = cfg['env']['projtype']
        
        # Ball Properties
        self.ball_size = cfg['env']['ballSize']
        self.ball_restitution = cfg['env']['ballRestitution']
        self.ball_density = cfg['env']['ballDensity']

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._build_target_tensors()

        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj_tensors()
        
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        self.obj_obs_size = 15
        obs_size += self.obj_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._proj_handles = []
            self._load_proj_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_target_asset(self): # smplx
        asset_root = "skillmimic/data/assets/urdf/" #projectname
        asset_file = "ball.urdf" #"ball.urdf"stick

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = self.ball_density #85.0#*6
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 1
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.disable_gravity = True
        # asset_options.fix_base_link = True

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ball = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Ball",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"skillmimic/data/assets/urdf/ball.urdf",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    angular_damping=0.01,
                    linear_damping=0.01,
                    #max_linear_velocity=1000.0,
                    max_angular_velocity=100.0,
                    #max_depenetration_velocity=100.0,
                    #enable_gyroscopic_forces=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(density = self.ball_density), #The mass of the rigid body (in kg)., The density of the rigid body (in kg/m^3).
                physics_material=sim_utils.RigidBodyMaterialCfg( # or maybe outside UsdFileCfg!!!!!!!!!!
                    #static_friction=1.0, 
                    #dynamic_friction=1.0,
                    restitution=self.ball_restitution, 
                )       
            )
            #init_state=ArticulationCfg.InitialStateCfg(
            #    pos=(0.0, 0, 0.4),
            #    rot=(0.1, 0.0, 0.0, 0.0),
            #    joint_pos={
            #        "door_left_joint": 0.0,
            #        "door_right_joint": 0.0,
            #        "drawer_bottom_joint": 0.0,
            #        "drawer_top_joint": 0.0,
            #    },
            #),
            
 
        )
        return
    
    def _load_proj_asset(self):
        asset_root = "skillmimic/data/assets/urdf/" #projectname
        small_asset_file = "block_projectile.urdf"
        
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)

        return
    
    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group, col_filter, segmentation_id)

        ball_props =  self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        # Modify the properties
        for b in ball_props:
            b.restitution = self.ball_restitution #0.66 #1.6
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, ball_props)  
        
        # set ball color
        if self.cfg["headless"] == False:
            self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(1.5, 1.5, 1.5))
                                        # gymapi.Vec3(0., 1.0, 1.5))
            h = self.gym.create_texture_from_file(self.sim, 'skillmimic/data/assets/urdf/basketball.png') #projectname
            self.gym.set_rigid_body_texture(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, h)


        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

        return
    
    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_PROJECTORS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)
            self.gym.set_actor_scale(env_ptr, proj_handle, 1.5)

        return
    
    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        self.init_obj_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float).repeat(self.num_envs, 1) 
        self.init_obj_rot_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        return

    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40

        num_actors = self.get_num_actors_per_env()
        num_objs = len(PERTURB_PROJECTORS)
        self._proj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - num_objs):, :]
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        self._calc_perturb_times()

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot") #ZC0
        self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")
        
        return

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_PROJECTORS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

        return


    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_target(self, env_ids):
        # self.init_obj_pos[env_ids, 2] += 8
        self._target_states[env_ids, :3] = self.init_obj_pos[env_ids]#.clone()+0.5
        self._target_states[env_ids, 3:7] = self.init_obj_rot[env_ids]#.clone() #rand_rot
        self._target_states[env_ids, 7:10] = self.init_obj_pos_vel[env_ids]#.clone()
        self._target_states[env_ids, 10:13] = self.init_obj_rot_vel[env_ids]
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.evts = list(self.gym.query_viewer_action_events(self.viewer))
        return 
    
    def post_physics_step(self):
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._update_proj()

        super().post_physics_step()

    def _update_proj(self):

        if self.projtype == 'Auto':
            curr_timestep = self.progress_buf.cpu().numpy()[0]
            curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
            perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]
            
            if (len(perturb_step) > 0):
                perturb_id = perturb_step[0]
                n = self.num_envs
                humanoid_root_pos = self._humanoid_root_states[..., 0:3]

                rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
                rand_theta *= 2 * np.pi
                rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_dist_min
                pos_x = rand_dist * torch.cos(rand_theta)
                pos_y = -rand_dist * torch.sin(rand_theta)
                pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_h_min
                
                self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + pos_x
                self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + pos_y
                self._proj_states[..., perturb_id, 2] = pos_z
                self._proj_states[..., perturb_id, 3:6] = 0.0
                self._proj_states[..., perturb_id, 6] = 1.0
                
                tar_body_idx = np.random.randint(self.num_bodies)
                tar_body_idx = 1

                launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
                launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
                launch_dir += 0.1 * torch.randn_like(launch_dir)
                launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
                launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
                launch_vel = launch_speed * launch_dir
                launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
                self._proj_states[..., perturb_id, 7:10] = launch_vel
                self._proj_states[..., perturb_id, 10:13] = 0.0

                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                             gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                             len(self._proj_actor_ids))
            
        elif self.projtype == 'Mouse':
            # mouse control
            for evt in self.evts:
                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(self.sim, self._proj_states, gymapi.STATE_ALL)
                elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
                    if evt.action == "mouse_shoot":
                        pos = self.gym.get_viewer_mouse_position(self.viewer)
                        window_size = self.gym.get_viewer_size(self.viewer)
                        xcoord = round(pos.x * window_size.x)
                        ycoord = round(pos.y * window_size.y)
                        print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

                    cam_pose = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

                    spawn = cam_pose.p
                    speed = 25
                    vel = cam_fwd * speed

                    angvel = 1.57 - 3.14 * np.random.random(3)

                    self._proj_states[..., 0] = spawn.x
                    self._proj_states[..., 1] = spawn.y
                    self._proj_states[..., 2] = spawn.z
                    self._proj_states[..., 7] = vel.x
                    self._proj_states[..., 8] = vel.y
                    self._proj_states[..., 9] = vel.z

                    self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                            gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                            len(self._proj_actor_ids))

        return

    
    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): 
            self.obs_buf[:] = obs

        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_obj_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = compute_obj_observations(root_states, tar_states)
        return obs



#####################################################################
###=========================jit functions=========================###
#####################################################################

################################################
@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat
###############################################

@torch.jit.script
def compute_obj_observations(root_states, tar_states):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    # for disturbance test
    # local_tar_pos += torch.rand_like(local_tar_pos).to(self.device)*0.05
    # local_tar_rot_obs += torch.rand_like(local_tar_rot_obs).to(self.device)*0.5
    # local_tar_vel += torch.rand_like(local_tar_vel).to(self.device)*0.5
    # local_tar_ang_vel += torch.rand_like(local_tar_ang_vel).to(self.device)*0.5

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs
