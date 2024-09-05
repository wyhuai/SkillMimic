import bpy
import math
import os
import pickle
import numpy as np

################# Setting parameters #################
obj_ind = 0 # don't change
interval = 30 # Select the frame interval to display motion
project_path = "/home/runyi/blender_for_SkillMimic"
task_name = f'demo_circling'
task_path = f'{project_path}/assets/skillmimic/{task_name}'
color_init = (0.134, 0.570, 0.014, 1) # display the humanoid in green color
location_offset = (0, 0.0, 0) # location offset for the humanoid and basketball


################# Step1: set the scene #################
# Delete all existing mesh objects
for obj in bpy.data.objects:
   bpy.data.objects.remove(obj, do_unlink=True)
# Delete the existing camera
for cam in bpy.data.cameras:
   bpy.data.cameras.remove(cam, do_unlink=True)

# Add playground
fbx_path = f"{project_path}/playground/006.fbx"
bpy.ops.import_scene.fbx(filepath=fbx_path)
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
for i, object in enumerate(imported_objects):
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
obj = imported_objects[0]
bpy.context.view_layer.objects.active = obj # 设置活动对象为第一个选中的对象
bpy.ops.object.join() # 合并
obj.name = "playground"
obj.scale = (0.033, 0.033, 0.033)
obj.rotation_euler = (0,0,-math.radians(90))
obj.location = (0,0,-0.02)
# Add a new camera  
bpy.ops.object.camera_add(location=(0, 0, 48))
camera = bpy.context.active_object
bpy.context.scene.camera = camera
camera.data.lens = 35.26
# Add a new light
bpy.ops.object.light_add(type='AREA', location=(0, 0, 7))
light = bpy.context.active_object
light.data.energy = 3000
light.data.size = 10
light.scale = (3.0, 2.0, 1.0)


################# Step2: set frame motion #################
select_frames = [_ for _ in range(0, len(os.listdir(task_path)), interval)]
with open(f'{project_path}/{task_name}.pickle', 'rb') as file:
   source_motion = pickle.load(file)

for i, frame_to_show in enumerate(select_frames):
   # load the humanoid frame motion
   bpy.ops.import_scene.obj(filepath=f"{task_path}/full_body{str(frame_to_show)}.obj")
   human_obj = bpy.context.selected_objects[0]
   human_obj.rotation_euler = (0,0,0) # (math.radians(90), 0, math.radians(90))
   human_obj.location = location_offset
   human_obj.scale = (1.0, 1.0, 1.0)
   mesh_name = "Default OBJ."+str(obj_ind).zfill(3) if i != 0 else "Default OBJ"
   bpy.data.materials[mesh_name].node_tree.nodes["Principled BSDF"].inputs[0].default_value = color_init
   obj_ind += 1
   # load the basketball frame motion
   ## make ball mesh
   bpy.ops.mesh.primitive_uv_sphere_add(
      radius=0.12, # 设置球体半径
      enter_editmode=False, # 以对象模式创建
      align='WORLD', # 对齐到世界坐标系
      location=(0, 0, 0) # 设置球体的位置
   )
   sphere = bpy.context.object
   sphere.name = f'Sphere_{frame_to_show}'
   ## set ball texture
   material = bpy.data.materials.new(name="BasketballMaterial")
   material.use_nodes = True
   nodes = material.node_tree.nodes
   links = material.node_tree.links
   nodes.clear()
   texture_node = nodes.new(type='ShaderNodeTexImage')
   texture_node.image = bpy.data.images.load(f"{project_path}/playground/basketball_uv_map.png")
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
   ## set ball pos & rot
   ball_pos = source_motion['ballpos']
   ball_rot = source_motion['ballrot']
   sphere.location = ball_pos[frame_to_show] + np.array(location_offset)
   sphere.rotation_mode = "QUATERNION"
   sphere.rotation_quaternion = ball_rot[frame_to_show]
   bpy.ops.object.shade_smooth()
   bpy.context.view_layer.update()
