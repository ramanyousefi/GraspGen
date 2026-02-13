### Group 3 Grasp Generation

This readme file explains how the demo and experiment for Group 3's project 3-1 were performed and how they can be repeated.

## GraspGen
One of the libraries used was GraspGen, a new grasp generation library that can function with both object meshes and scene point clouds.
The final implementation is based on the demo scripts included in GraspGen
The Demo assumes the object meshes for the slides are inside a Meshes folder inside the GraspGen folder, but they use and absolute path, so you have to make sure to correct this when trying to recreate it. The path to the gripper yaml file is also an absolute path that might need to be changed.

# Installation
To install GraspGen, follow the instructions on their github: https://github.com/NVlabs/GraspGen

# Experiment
For the experiment we slightly modified their demo_object_mesh.py to print the results of the grasp generation.
We used this to generated a total of 150 grasps on the Slider_1.stl mesh in 2 batches, one of 50 and one of 100 grasps.

# Demo
The Demo folder includes everything needed to run the demo presented during the business pitch

Some stepts that show our implementation and the functionality of the library:
1. Open the conda environment which was created for GraspGen
2. Run the command: meshcat-server
3. Copy the https link and paste it into a browser, 
    it should open to a mostly empty space that you can look around in
4. Run the command: python GraspGenScript.py --server, this might need some path adjustment depending on where everything is stored
    This should print some grasps as output, but will not change anything in the browser
5. Move to the main GraspGen folder, the one containing the scripts folder
6. Run the command: python scripts/demo_object_mesh.py --mesh_file Meshes/Slider_1.stl --mesh_scale 0.001 --gripper_config models/checkpoints/graspgen_robotiq_2f_140.yml
    Again, check to make sure the pathing is correct
    Once run, this should change the browser page to now have a mesh of a slider surrounded by grasps
    You can use the mouse to look around and rotate the slide to get a better look
7. Now run the command: python scripts/demo_scene_pc.py --sample_data_dir models/sample_data/real_scene_pc --gripper_config models/checkpoints/graspgen_robotiq_2f_140.yml
    This should change the browser page to now show a point cloud of a scene, with grasps around a single object




## VGN
For our implementation we slightly modified clutter_removal.py and simulation.py, the modified versions will be included in the same folder as this readme, but need to be placed in the correct location when trying to recreate our implementation. Those locations are:
src/vgn/experiments/clutter_removal.py
src/vgn/simulaiton.py

Different versions of the slides had to be created to fit into VGN, this conserns the following files:
slide.obj
slide.urdf
Slide2.obj
slide2.urdf
These files are currently stored in the Meshes folder delivered alongside this readme, but need to be stored in the data/urdfs/slide/ folder.

# Installation
To install GraVGNspGen, follow the instructions on their github: https://github.com/ethz-asl/vgn 

# Experiment
The original slider meshes were in milimiters, whereas vgn works with meters, so make sure to have a correctly scaled version.
python scripts/sim_grasp.py --model data/models/vgn_conv.pth --object-set slide --num-objects 1 --num-rounds 1 --sim-gui --rviz
The experiment was only run on slide 1.

## Simple grasps
Simple grasps contains a seperate readme file inside the Simple Grasps folder.



## Evaluation
For the evaluation make sure you have the pybullet library installed.
The script used to evaluate grasps, as well as the text files containing the grasps we used in our experiment are inside the Evaluation folder.

