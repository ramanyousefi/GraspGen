# PLY-TO-POINTCLOUD2 CONVERTER AND GRASP GENERATOR

## Overview

This repository contains two Python scripts for testing and integrating the `simple_grasping` ROS grasp planner using point cloud data. The scripts support grasp generation from real object meshes stored in `.ply` format, as well as from synthetic objects used for testing and validation.

The script `test_grasp_planning.py` is intended for verifying that all required ROS nodes are running correctly and that the `simple_grasping` pipeline is functioning as expected.  
The script `ply_to_ptcld.py` is used to generate grasp candidates for real objects by converting a `.ply` mesh into a `PointCloud2` message and passing it to the grasp planner.

---

## Dependencies

- ROS (ROS1)
- `simple_grasping` (https://github.com/mikeferguson/simple_grasping/blob/ros2/README.md?plain=1)
- `grasping_msgs`
- `actionlib`
- `sensor_msgs`
- `visualization_msgs`
- `numpy`

---

## Usage

### Test Grasp Planning with Synthetic Objects
```
rosrun <your_package> test_grasp_planning.py
```
This script publishes synthetic point clouds (e.g., cylinders and spheres), requests grasp planning, and visualizes the resulting grasps in RViz.

### Generate Grasps from a PLY Mesh
```
rosrun <your_package> ply_to_ptcld.py path/to/object.ply --frame-id base_link --scale 1.0

```
* **ply_file:** Path to the input .ply mesh file
* **--frame-id:** TF frame used for the point cloud (default: base_link)
* **--scale:** Uniform scaling factor applied to the object (default: 1.0)

---

## Notes

* The simple_grasping library is using ROS1 branch, and the default is ROS2. Please be careful to switch to ROS1 branch after cloning the repository.

* Grasps are generated for parallel-jaw grippers, which are required by simple_grasping.

* Thin objects may be ignored by the grasp planner; the mesh is artificially thickened during preprocessing when necessary.

* A table plane is intentionally included in the point cloud so the segmentation stage can identify and remove it.

* For visualization, RViz should subscribe to:

/head_camera/depth_registered/points (PointCloud2)

/grasp_markers (MarkerArray)

