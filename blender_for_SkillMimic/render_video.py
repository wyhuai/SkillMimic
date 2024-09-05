import sys
sys.path.append(r"C:\Users\11400\appdata\roaming\python\python310\site-packages")
import os
import bpy
import math
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

################# Setting parameters #################
JOINT = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Spine2', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3'] # do not change
obj_ind = 0 # don't change
interval = 30 # Select the frame interval to display motion
project_path = "/home/runyi/blender_for_SkillMimic"
task_name = f'demo_circling'
task_path = f'{project_path}/{task_name}'
color_init = (0.134, 0.570, 0.014, 1) # display the humanoid in green color
motion_scale = 0.5
location_offset = (0, 0.0, 0) # location offset for the humanoid and basketball

############## Step1: set the scene #################
# Delete all existing mesh objects
for obj in bpy.data.objects:
   bpy.data.objects.remove(obj, do_unlink=True)
# Delete the existing camera
for cam in bpy.data.cameras:
  bpy.data.cameras.remove(cam, do_unlink=True)

# Add playground
bpy.ops.import_scene.fbx(filepath=f'{project_path}/playground/006.fbx')
imported_objects = bpy.context.selected_objects
# Add material for the playground
texture_mapping = {
  'archmodels81_006_01': 'archmodels81_006_006.jpg',
  'archmodels81_006_02': None,
  'archmodels81_006_03': None,
  'archmodels81_006_04': None,
  'archmodels81_006_05': 'archmodels81_006_001.jpg',
  'archmodels81_006_06': 'archmodels81_006_007.jpg',
  'archmodels81_006_07': 'archmodels81_006_008.jpg',
  'archmodels81_006_08': None,
}

indices = [0]
for i, object in enumerate(imported_objects):
   if i not in indices:
       bpy.context.view_layer.objects.active = object
       object.select_set(True)
       for other_obj in bpy.context.selected_objects:
           if other_obj != object:
               other_obj.select_set(False)
       bpy.ops.object.delete()
       continue
   if texture_mapping[object.name] is None:
       continue
   material = bpy.data.materials.new(name=f"{object.name}")
   material.use_nodes = True
   nodes = material.node_tree.nodes
   links = material.node_tree.links
   nodes.clear()
   texture_node = nodes.new(type='ShaderNodeTexImage')
   texture_node.image = bpy.data.images.load(f"{project_path}/playground/{texture_mapping[object.name]}")
   bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
   output_node = nodes.new(type='ShaderNodeOutputMaterial')
   texture_node.location = (-300, 0)
   bsdf_node.location = (0, 0)
   output_node.location = (300, 0)
   links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])
   links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
   if object.data.materials:
       object.data.materials[0] = material
   else:
       object.data.materials.append(material)
# playground scale & rotation
playground_objects = [bpy.data.objects[list(texture_mapping.keys())[ind]] for ind in indices]
for obj in playground_objects:
   # bpy.context.collection.objects.link(obj)
   obj.select_set(True)

bpy.context.view_layer.objects.active = playground_objects[0] # 设置活动对象为第一个选中的对象
bpy.ops.object.join() # 合并
playground_objects[0].name = "playground"
playground_objects[0].scale = (0.011, 0.011, 0.011)
playground_objects[0].rotation_euler = (0, 0, 0) # (0,0,-math.radians(90))
playground_objects[0].location = (0,0,-0.02)

# Add a new camera  
bpy.ops.object.camera_add(location=(-6, 0, 2))
camera = bpy.context.active_object
bpy.context.scene.camera = camera
camera.data.lens = 56.09
camera = bpy.data.objects['Camera']
camera.constraints.new(type='TRACK_TO')
# Add empty cube
bpy.ops.object.empty_add(type='CUBE')
cube_empty = bpy.context.active_object
cube_empty.location = (0, 0, 0.5) 
camera.constraints['Track To'].target = cube_empty

bpy.ops.object.light_add(type='AREA', location=(0, 0, 7))
light = bpy.context.active_object
light.data.energy = 5000
light.data.size = 10
light.scale = (3.0, 2.0, 1.0)

# make ball mesh
bpy.ops.mesh.primitive_uv_sphere_add(
   radius=0.06, # 设置球体半径
   enter_editmode=False, # 以对象模式创建
   align='WORLD', # 对齐到世界坐标系
   location=(0, 0, 0) # 设置球体的位置
)
sphere = bpy.context.object
bpy.ops.object.shade_smooth()
# set ball texture
material = bpy.data.materials.new(name="BasketballMaterial")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
nodes.clear()
texture_node = nodes.new(type='ShaderNodeTexImage')
texture_node.image = bpy.data.images.load(f"{project_path}/playground/basketball.png")
bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
output_node = nodes.new(type='ShaderNodeOutputMaterial')
texture_node.location = (-300, 0)
bsdf_node.location = (0, 0)
output_node.location = (300, 0)
links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])
links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
if sphere.data.materials:
   sphere.data.materials[0] = material
else:
   sphere.data.materials.append(material)


################# Step2: load joint obj #################
obj_files = f'{project_path}/body_mesh'
for obj_name in os.listdir(obj_files):
   bpy.ops.import_scene.obj(filepath=os.path.join(obj_files, obj_name))
for obj in bpy.data.objects:
    joint_name = obj.name if '.' not in obj.name else obj.name.split('.')[0]
    if joint_name in JOINT:
        obj.scale = (motion_scale, motion_scale, motion_scale)
        mesh_name = "Default OBJ."+str(obj_ind).zfill(3) if obj_ind != 0 else "Default OBJ"
        bpy.data.materials[mesh_name].node_tree.nodes["Principled BSDF"].inputs[0].default_value = color_init
        obj_ind += 1


################# Step3: set frame motion #################
scene = bpy.context.scene
current_frame = scene.frame_current
with open(f'{project_path}/{task_name}.pickle', 'rb') as file:
   source_motion = pickle.load(file)

# load the humanoid motion
obj = bpy.data.objects['Pelvis']
for frame in range(1, 100):
    scene.frame_set(frame)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if obj.name in JOINT:
                bone_ind = JOINT.index(obj.name)
                obj.rotation_euler = R.from_quat(source_motion['dofrot'][frame,bone_ind]).as_euler('xyz')
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                obj.location = (source_motion['dofpos'][frame,bone_ind] * motion_scale) + location_offset
                obj.keyframe_insert(data_path="location", frame=frame)
            if obj.name == 'Sphere':
                obj.rotation_euler = R.from_quat(source_motion['ballrot'][frame]).as_euler('xyz')
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                obj.location = source_motion['ballpos'][frame] * motion_scale + location_offset
                obj.keyframe_insert(data_path="location", frame=frame)
    bpy.context.view_layer.update()
