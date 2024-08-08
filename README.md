# SkillMimic
[Paper](https://github.com/wyhuai/SkillMimic) | [Project Page](https://github.com/wyhuai/SkillMimic) | [Video](https://github.com/wyhuai/SkillMimic)

Official code release for the following paper:
"**SkillMimic: Learning Reusable Basketball Skills from Demonstrations**"

![image](https://github.com/user-attachments/assets/ac75c9be-f144-4b6d-980f-272c6f657627)
We propose a novel approach that enables physically simulated humanoids to learn a variety of basketball skills purely from human demonstrations, such as
shooting (blue), retrieving (red), and turnaround layup (yellow). Once acquired, these skills can be reused and combined to accomplish complex tasks, such as
continuous scoring (green), which involves dribbling toward the basket, timing the dribble and layup to score, retrieving the rebound, and repeating the whole process.

## TODOs

- [ ] Release the complete raw BallPlay-M dataset and the data processing code.

- [x] Release a subset of the BallPlay-M dataset.

- [x] Release training and evaluation code.

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


## Pre-Trained Models
Download the pre-trained models from here [link](https://??), unzip the files, and put them into `skillmimic/data/models/`. The directory structure should be like `skillmimic/data/models/mixedskills/nn/PhysHOI.pth`, etc.

## Skill Policy ⛹️‍♂️
The skill policy can be trained purely from demonstrations, without the need for designing case-by-case skill rewards. Our method allows a single policy to learn a large variety of basketball skills from a dataset that contains diverse skills. 
### Inference
Run the following command.
```
python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 16 --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/[task].pt --checkpoint physhoi/data/models/[task]/nn/PhysHOI.pth
```
- You may control the skill switching using your keyboard. By default, the key and skill correspondence are as follows:
`1: pick up`, `2: shot`, `3: dribble left`, `4: dribble forward`, `5: dribble right`, `6: layup`, `7: turnaround layup`.
- You may change `--motion_file` to alter the initialization, or add `--init_start_frame` to disable random initialization.
- To view the HOI dataset, add `--play_dataset`.
- To save the images, add `--save_images test_images` to the command, and the images will be saved in `physhoi/data/images/test_images`.
- To transform the images into a video, run the following command, and the video can be found in `physhoi/data/videos`.
```
python physhoi/utils/make_video.py --image_path physhoi/data/images/test_images --fps 60
```

### Training
To train the skill policy, run the following command: 
```
python physhoi/run.py --task PhysHOI_BallPlay --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/toss.pt --headless
```
- During the training, the latest checkpoint PhysHOI.pth will be regularly saved to output/, along with a Tensorboard log.
- You may change the `--motion_file` to train skill policy on different data, e.g., `--motion_file physhoi/data/motions/skillset_1`.
- It is strongly encouraged to use large "--num_envs" when training on a large dataset, e.g., use "--num_envs 16384" for `--motion_file physhoi/data/motions/skillset_1`

## High-Level Controller ⛹️‍♂️
Once the skill policy is learned, we can train a high-level controller to reuse the learned skills to accomplish complex high-level tasks.

### Inference
Run the following command.
```
python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 16 --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/[task].pt --checkpoint physhoi/data/models/[task]/nn/PhysHOI.pth
```
- You may change the target position by clicking your mouse.

### Training
To train the skill policy, run the following command: 
```
python physhoi/run.py --task PhysHOI_BallPlay --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/toss.pt --headless
```

### The BallPlay dataset 🏀

## References
If you find this repository useful for your research, please cite the following work.
```
@article{wang2023physhoi,
  author    = {Wang, Yinhuai and Lin, Jing and Zeng, Ailing and Luo, Zhengyi and Zhang, Jian and Zhang, Lei},
  title     = {PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction},
  journal   = {arXiv preprint arXiv:2312.04393},
  year      = {2023},
}
```
The code implementation is based on ASE and PhysHOI:
- https://github.com/nv-tlabs/ASE
- https://github.com/wyhuai/PhysHOI

The SMPL-X humanoid robot is generated using UHC:
- https://github.com/ZhengyiLuo/UniversalHumanoidControl


