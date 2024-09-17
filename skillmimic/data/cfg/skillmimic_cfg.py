from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg

@configclass
class SkillmimiceEnvCfg(DirectRLEnvCfg):
   # env
   decimation = 2 # same as substeps in issacgym
   episode_length_s = 5.0
   num_actions = 1
   num_observations = 4
   num_states = 0

   # simulation
   sim: SimulationCfg = SimulationCfg(
   #device = "cuda:0" # can be "cpu", "cuda", "cuda:<device_id>"
   #dt=1 / 120,
   # decimation will be set in the task config
   # up axis will always be Z in isaac sim
   # use_gpu_pipeline is deduced from the device
   # gravity=(0.0, 0.0, -9.81),
   physx: PhysxCfg = PhysxCfg(
       # num_threads is no longer needed
       solver_type=1,
       # use_gpu is deduced from the device
       max_position_iteration_count=4,
       max_velocity_iteration_count=0,
       # moved to actor config contact_offset, rest_offset, max_depenetration_velocity
       bounce_threshold_velocity=0.2,
       # default_buffer_size_multiplier is no longer needed
       # gpu_max_rigid_contact_count=2**23 # same as max_gpu_contact_pairs in isaac gym, not set in original cfg file
   ))
   # robot
   robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"skillmimic/data/assets/mjcf/mocap_humanoid_boxhand.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # ball


   # scene
   scene: InteractiveSceneCfg = InteractiveSceneCfg(
   num_envs=2048,
   env_spacing=2.0)



   # ground plane
   terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1, #???
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # friction_combine_mode="multiply",
            # restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8, 
        ),
    )
   
   env =  {
    "rewardWeights" : {
        "p": 20.0,
        "r": 20.0,
        "pv": 0.0,
        "rv": 0.0,

        "op": 1.0,
        "or": 0.0,
        "opv": 0.0,
        "orv": 0.0,

        "ig": 20.0,

        "cg1": 5.0,
        "cg2": 5.0,
        },

    "enableTaskObs" : False, #ZC0

    "enableDebugVis": False,
    "playdataset": None,
    "projtype": "None",
    "saveImages": "",
    "initVel": False,

    "pdControl": True,
    "powerScale": 1.0,
    "controlFrequencyInv": 1,  # 60 Hz
    "stateInit": "Random",
    "hybridInitProb": 0.5,
    "dataFPS": 120,
    "dataFramesScale": 0.5,

    "ballSize": 1. ,
    "ballRestitution": 0.81,
    "ballDensity": 1000.

    }
    
