# SkillMimic
[Paper](https://arxiv.org/abs/2408.15270) | [Project Page](https://ingrid789.github.io/SkillMimic/) | [Video](https://youtu.be/j1smsXilUGM)

Official code release for the following paper:
"**SkillMimic: Learning Reusable Basketball Skills from Demonstrations**"

![image](https://github.com/user-attachments/assets/ac75c9be-f144-4b6d-980f-272c6f657627)
We propose a novel approach that enables physically simulated humanoids to learn a variety of basketball skills purely from human demonstrations, such as
shooting (blue), retrieving (red), and turnaround layup (yellow). Once acquired, these skills can be reused and combined to accomplish complex tasks, such as
continuous scoring (green), which involves dribbling toward the basket, timing the dribble and layup to score, retrieving the rebound, and repeating the whole process.

## TODOs

- [ ] Release the complete raw BallPlay-M dataset and the data processing code.

- [X] Release a subset of the BallPlay-M dataset.

- [X] Release training and evaluation code.

## Installation 💽

### Step 1: create conda environment
```
conda create -n skillmimic python=3.8
conda activate skillmimic
pip install -r requirements.txt
```
Or you can simply run the following command
```
conda env create -f environment.yml
```

### Step 2: download and install the Issac Gym

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
unzip the file using following command.

```
tar -xzvf /{your_source_dir}/IsaacGym_Preview_4_Package.tar.gz -C /{your_target_dir}/
cd /{your_target_dir}/isaacgym/python/
pip install -e .
```

Test the Isaac Gym.
```
cd /{your_target_dir}/isaacgym/python/examples
python joint_monkey.py
```

If you can see the pop-up interface, it means the installation is successful.

However, if you encounter the message `ImportError: libpython3.*m.so.1.0: cannot open shared object ...`, don't worry. This is a common issue and can be easily resolved.

You might take [this](https://github.com/google-deepmind/acme/issues/47 ) for your reference.


## Pre-Trained Models
Pre-trained models are available at `skillmimic/data/models/`

## Skill Policy ⛹️‍♂️
The skill policy can be trained purely from demonstrations, without the need for designing case-by-case skill rewards. Our method allows a single policy to learn a large variety of basketball skills from a dataset that contains diverse skills. 

### Inference
Run the following command.
```
python skillmimic/run.py --test --task SkillMimicBallPlay --num_envs 16 \
--cfg_env skillmimic/data/cfg/skillmimic.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
--motion_file skillmimic/data/motions/BallPlay-M/layup \
--checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```
- You may control the skill switching using your keyboard. By default, the key and skill correspondence are as follows:
`Q: pick up`, `W: shot`, `←: dribble left`, `↑: dribble forward`, `→: dribble right`, `E: layup`, `R: turnaround layup`.
- You may change `--motion_file` to alter the initialization, or add `--state_init frame_number` to initialize from a specific reference state (Default: random reference state initialization).
- To view the HOI dataset, add `--play_dataset`.
- To save the images, add `--save_images test_images` to the command, and the images will be saved in `skillmimic/data/images/test_images`.
- To transform the images into a video, run the following command, and the video can be found in `skillmimic/data/videos`.
```
python skillmimic/utils/make_video.py --image_path skillmimic/data/images/test_images --fps 60
```

### Training
To train the skill policy, run the following command: 
```
python skillmimic/run.py --task SkillMimicBallPlay \
--cfg_env skillmimic/data/cfg/skillmimic.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
--motion_file skillmimic/data/motions/BallPlay-M/layup --headless
```
- During the training, the latest checkpoint SkillMimic.pth will be regularly saved to output/, along with a Tensorboard log.
- `--cfg_env` specifies the environment configurations, such as number of environments, dataFPS, ball properties, etc.
- `--cfg_train` specifies the training configurations, such as learning rate, maximum number of epochs, network settings, etc.
- `--motion_file` can be changed to train on different data, e.g., `--motion_file skillmimic/data/motions/BallPlay-M/skillset_1`.
- `--headless` is used to disable visualization.
- It is strongly encouraged to use large "--num_envs" when training on a large dataset, e.g., use "--num_envs 16384" for `--motion_file skillmimic/data/motions/skillset_1` (Meanwhile, `--minibatch_size` is recommended to be set as 8×`num_envs`)


## High-Level Controller ⛹️‍♂️
Once the skill policy is learned, we can train a high-level controller to reuse the learned skills to accomplish complex high-level tasks.

### Inference
The testing command of high-level tasks is in the following format: 
```
python skillmimic/run.py --test --task [HRLTaskName] --num_envs 16 \
--cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/[configFile] \
--motion_file skillmimic/data/motions/BallPlay-M/[task] \
--checkpoint skillmimic/data/models/[task]/nn/SkillMimic.pth \
--llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```
- You may change the target position by clicking your mouse by adding `--projtype Mouse`.
  
Here are specific commands for testing 4 tasks:

Circling:
```
python skillmimic/run.py --test --task HRLCircling --num_envs 1 --projtype Mouse --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_circling.yaml --motion_file skillmimic/data/motions/BallPlay-M/run --checkpoint skillmimic/data/models/hlc_circling/nn/SkillMimic.pth --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

Heading:
```
python skillmimic/run.py --test --task HRLHeadingEasy --num_envs 1 --projtype Mouse --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_heading.yaml --motion_file skillmimic/data/motions/BallPlay-M/run --checkpoint skillmimic/data/models/hlc_heading/nn/SkillMimic.pth --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

Throwing:
```
python skillmimic/run.py --test --task HRLThrowing --num_envs 1 --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_throwing.yaml --motion_file skillmimic/data/motions/BallPlay-M/turnhook --checkpoint skillmimic/data/models/hlc_throwing/nn/SkillMimic.pth --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

Scoring
```
python skillmimic/run.py --test --task HRLScoringLayup --num_envs 1 --projtype Mouse --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_layupscore.yaml --motion_file skillmimic/data/motions/BallPlay-M/run --checkpoint skillmimic/data/models/hlc_scoring/nn/SkillMimic.pth --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

### Training
The training command of high-level tasks is in the following format: 
```
python skillmimic/run.py --task [HRLTaskName] \
--cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/[configFile] \
--motion_file skillmimic/data/motions/BallPlay-M/[task] \
--llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth --headless
```
- `--llc_checkpoint` specifies the checkpoint for the low-level controller. A pre-trained low-level controller is available in `skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth`
  
Here are specific commands for training 4 tasks:

Circling:
```
python skillmimic/run.py --task HRLCircling --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_circling.yaml --motion_file skillmimic/data/motions/BallPlay-M/run --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth --headless
```

Heading:
```
python skillmimic/run.py --task HRLHeadingEasy --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_heading.yaml --motion_file skillmimic/data/motions/BallPlay-M/run --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth --headless
```

Throwing:
```
python skillmimic/run.py --task HRLThrowing --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_throwing.yaml --motion_file skillmimic/data/motions/BallPlay-M/turnhook --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth --headless
```

Scoring:
```
python skillmimic/run.py --task HRLScoringLayup --cfg_env skillmimic/data/cfg/skillmimic_hlc.yaml --cfg_train skillmimic/data/cfg/train/rlg/hrl_humanoid_discrete_layupscore.yaml --motion_file skillmimic/data/motions/BallPlay-M/run --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth --headless
```

## The BallPlay dataset 🏀
A subset is at `skillmimic/data/motions/BallPlay-M/`, full dataset coming soon.


## References 🔗
If you find this repository useful for your research, please cite the following work.
```
@article{wang2024skillmimic,
author    = {Wang, Yinhuai and Zhao, Qihan and Yu, Runyi and Zeng, Ailing and Lin, Jing and Luo, Zhengyi and Tsui, Hok Wai and Yu, Jiwen and Li, Xiu and Chen, Qifeng and Zhang, Jian and Zhang, Lei and Tan Ping},
  title     = {SkillMimic: Learning Reusable Basketball Skills from Demonstrations},
  journal   = {arXiv preprint arXiv:2408.15270},
  year      = {2024},
}
```

## Render 🎨
See the [folder](./blender_for_SkillMimic)

## Acknowledgements 👏
The code implementation is based on ASE and PhysHOI:
- https://github.com/nv-tlabs/ASE
- https://github.com/wyhuai/PhysHOI

The rendering implementation is based on UniHSI:
- https://github.com/OpenRobotLab/UniHSI
- https://github.com/xizaoqu/blender_for_UniHSI