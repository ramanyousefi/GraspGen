
# Zivid Vision to GraspGen ROS Pipeline

This repository contains an automated pipeline designed to take raw 3D point cloud data (specifically from a Zivid vision camera) and generate collision-free, robot-ready grasp poses using NVIDIA's GraspGen framework.

The pipeline handles the entire workflow: point cloud normalization, object segmentation, grasp inference, collision filtering, and final export to MoveIt/ROS formats.

## 🚀 Quickstart: How to Replicate the Results

To get the exact same results in your environment, follow these steps:

### 1. Prerequisites

Ensure you have the following installed in your Python environment:

* Python 3.8+
* `open3d` (for point cloud processing)
* `numpy`, `scipy`, `trimesh`, `omegaconf`, `torch`
* NVIDIA `grasp_gen` library (Ensure the `grasp_gen.grasp_server` is accessible)

### 2. Configure Your Workspace

Before running the pipeline, you **must** configure the physical boundaries of your workspace.

* Open `convert_ply_to_json.py`.
* Modify the `CROP_COORDS` dictionary and `OBJECT_HEIGHT` to match the exact millimeter coordinates of your target object's bounding box in the camera's view.

### 3. Run the Pipeline

Place your raw input point cloud (e.g., `1-slider-in-box.ply`) in the root directory and run the orchestrator script:

```bash
python3 run_pipeline.py 1-slider-in-box.ply

```

---

## 📂 Output Files

Once the pipeline runs successfully, you will find the final exported grasps in the `graspgen_data/exported_grasps` directory:

* `*_moveit_grasps.json`: Poses formatted for geometry_msgs/Pose (quaternions).
* `*_robot_grasps.json`: Poses formatted as raw SE(3) transformation matrices.

---

## 🛠️ What the Code Does (Pipeline Architecture)

The pipeline is orchestrated by `run_pipeline.py`, which executes three main steps sequentially:

### Step 1: Normalization (`flip_ply.py`)

Raw camera point clouds often have inverted coordinate systems. This script loads the input `.ply` file and inverts a specified axis (default is Z) to normalize the orientation of the point cloud.

* **Input**: `scan.ply`
* **Output**: `fixed.ply`

### Step 2: Segmentation & Conversion (`convert_ply_to_json.py`)

This script prepares the data for the GraspGen neural network.

* **Segmentation**: It isolates the target object from the background using the hardcoded `CROP_COORDS` and defines a "scene" by adding padding around the object.
* **Scaling**: It scales the point cloud from millimeters to meters using a 0.001 scale factor.
* **Formatting**: It saves the processed object points, scene colors, and object masks into a standardized JSON format.

### Step 3: Inference & Collision Filtering (`my_scene_pc.py`)

This is the core grasp generation script.

* **Inference**: It loads the JSON and runs GraspGen to sample 2,000 potential grasp poses.
* **Collision Detection**: It rigorously filters the generated grasps against the scene point cloud using the actual collision mesh of your specific gripper (e.g., `graspgen_robotiq_2f_140.yml`).
* **Visualization & Export**: It centers the point cloud for Meshcat visualization and converts the collision-free grasps back to the original world frame before exporting them.

---

## ⚙️ Advanced Settings

* **Changing the Gripper**: Update the `GRIPPER_CONFIG` path in `run_pipeline.py` to point to a different YAML configuration file.
* **Grasp Density**: In `run_pipeline.py`, you can change the `--num_grasps 2000` argument in the inference command to generate more or fewer grasps.
