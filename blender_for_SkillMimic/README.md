## Requirements
You can download the blender from [here](https://www.blender.org/download/). 
Remember to download a Blender version lower than 4.0, as some functions are deprecated in higher versions.

## Render images for selected frame motion
Step1: Get the body mesh for each frame & Get the motion file in pickle format
```
python motionseq2mesh.py
```
Step2: Copy the code from [render_figure.py](./render_figure.py) into Blender. 
Step3: Adjust the parameters and camera settings in Blender to render the figure you want.

## Render videos for motion sequence
Step1: Install the `scipy` package in Blender's Python environment
Step2: Get the motion file in pickle format
```
python motionseq2mesh.py
```
Step3: Copy the code from [render_video.py](./render_figure.py) into Blender. 
Step4: Adjust the parameters and camera settings in Blender to render the video.
