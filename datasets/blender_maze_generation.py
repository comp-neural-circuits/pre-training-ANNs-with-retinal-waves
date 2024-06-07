import bpy
from random import randint
import math

import config # contains the path to the project directory

total_nr_images = 500

for wall_left_nr in range(0,total_nr_images):
    cue_nr = randint(1,20)
    bpy.ops.import_image.to_plane(files=[{"name":f"cue_{cue_nr}.jpg"}], directory=f"{config.project_dir}/real_world_dataset/cues/JPEG/", relative=False)
    bpy.data.materials[f"cue_{cue_nr}"].node_tree.nodes["Image Texture"].interpolation = 'Closest'
    bpy.context.object.rotation_euler[2] = 3.14508
    bpy.context.object.location[0] = wall_left_nr +0.5
    bpy.context.object.location[1] = 0.7
    bpy.context.object.location[2] = 0.5

for wall_right_nr in range(0,total_nr_images):
    cue_nr = randint(1,20)
    bpy.ops.import_image.to_plane(files=[{"name":f"cue_{cue_nr}.jpg"}], directory=f"{config.project_dir}/real_world_dataset/cues/JPEG/", relative=False)
    bpy.data.materials[f"cue_{cue_nr}"].node_tree.nodes["Image Texture"].interpolation = 'Closest'
    bpy.context.object.rotation_euler[2] = 3.14508
    bpy.context.object.location[0] = wall_right_nr +0.5
    bpy.context.object.location[1] = -0.7
    bpy.context.object.location[2] = 0.5


#floor
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=((0.5)*total_nr_images, 0, -0.05), rotation=(0, -0, 0), scale=(total_nr_images, 1.4, 0.1))

#wall at the end
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(total_nr_images +0.05, 0, 0.5), rotation=(0, -0, 0), scale=(0.1, 1.4, 1))

#orient camera
cam = bpy.data.objects['Camera']
cam.location.x = -1
cam.location.y = 0
cam.location.z = 0.5
cam.rotation_euler[0] = 90 * (math.pi / 180.0)
cam.rotation_euler[1] = 0
cam.rotation_euler[2] = -90 * (math.pi / 180.0)

bpy.ops.curve.primitive_nurbs_path_add(radius=1, enter_editmode=False, align='WORLD', location=((total_nr_images/2), 0, 0.5), scale=(1, 1, 1))

bpy.context.space_data.context = 'WORLD'
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.522517, 0.522517, 0.522517, 1)


#TODO in blender:
# 1. set render settings: y=128, x=64
# 2. create path (shift + A, path, set size and position); then tab (go into object mode); click on first node; shift S; Cursor to selected; Select camere; shift S; Selection to cursor; select camera and path; cmd + P; follow path
# 3. adjust speed of camera: click on path, then object data properties on the right, then path animation and change number of frames
# 4. set light as child of camera (on the right, under "Object Constraint Properties")