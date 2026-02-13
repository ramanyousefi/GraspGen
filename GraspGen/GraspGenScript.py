import numpy as np
import json
from GraspGen.grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from scipy.spatial.transform import Rotation as R

class GraspGenModule:
    # Some absolute paths
    GRIPPER_PATH = "/home/moveit2/Documents/Group3_GraspGeneration/GraspGen/models/checkpoints/graspgen_robotiq_2f_140.yml"
    JSON_DATA_PATH = "/home/moveit2/Documents/Group3_GraspGeneration/GraspGen/models/sample_data/real_scene_pc/1745766797_642935.json"

    def __init__(self, gripper_config_path=GRIPPER_PATH):
        self.grasp_cfg = load_grasp_cfg(gripper_config_path) # Load gripper file
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg) # Load machine learning model

    def GenerateGrasps(self,
        point_cloud: np.ndarray, # Float array (N,3), point cloud of object
        _grasp_threshold: float = 0.8, # Confidence threshold
        _num_grasps: int = 200, # How many grasps to sample
        _topk_num_grasps: int = -1, # Return top k grasps if > 0
    ):       
        # Ensure correct data type
        obj_pc = np.asarray(point_cloud) 
        obj_pc = obj_pc.astype(np.float32)

        # Generate grasps
        grasps, scores = GraspGenSampler.run_inference(
            obj_pc,
            self.grasp_sampler,
            grasp_threshold=_grasp_threshold,
            num_grasps=_num_grasps,
            topk_num_grasps=_topk_num_grasps
        )

        # Turn PyTorch tensors into numpy arrays
        grasps = grasps.cpu().numpy()
        scores = scores.cpu().numpy()

        # Sort grasps by confidence level
        idx = np.argsort(scores)[::-1]
        grasps = grasps[idx]
        scores = scores[idx]

        return grasps, scores
   
    # Method that retrieves a JSON file to use as input
    def UseJSONInput(self, json_path=JSON_DATA_PATH) -> np.ndarray:
        # Load data in JSON file
        with open(json_path, "r") as f:
            data = json.load(f)
        if "object_info" in data and "pc" in data["object_info"]:
            obj_pc = np.asarray(data["object_info"]["pc"], dtype=np.float32)

        # Check if input is valid
        if obj_pc.ndim != 2 or obj_pc.shape[1] != 3:
            raise ValueError("scene_point_cloud must have shape (N, 3)")
        if len(obj_pc) < 50:
            raise ValueError("Point cloud too small for grasp generation")

        return obj_pc
   
   # Get object point cloud from request
    def RetrieveInput(self, request: dict) -> np.ndarray:
        pc = request["point_cloud"]
        mask = request["object_mask"]

        # Check if input is valid
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError("scene_point_cloud must have shape (N, 3)")
        if len(pc) < 50:
            raise ValueError("Point cloud too small for grasp generation")
        if pc.shape[0] != mask.shape[0]:
            raise ValueError("Point cloud and mask size mismatch")

        obj_pc = pc[mask]
        return obj_pc
   
   # Give output a consistent agreed upon format
    def OutputFormatting(self, 
        grasps: np.ndarray,
        scores: np.ndarray,
        frame="Base",
        default_width: float | None = None
    ):
        grasp_list = []

        for T, score in zip(grasps, scores):
            position = T[:3, 3] # Spational coordinates
            rotation_matrix = T[:3, :3]
            quat = R.from_matrix(rotation_matrix).as_quat() # Turn 3x3 rotational matrix into quaternions for easy of use

            grasp = {
                "position": position.tolist(),
                "rotation": quat.tolist(),
                "width": default_width, # How wide the gripper opens before grasping, not supported by GraspGen but included in the format
                "score": float(score),
                "frame": frame # Signals where the O(0,0) of the frame is
            }

            grasp_list.append(grasp)

        return grasp_list

# main function that starts a server that can be contacted by a ROS node
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    # Current hard-coded implementation that uses a JSON file instead of input from a ROS node
    gm = GraspGenModule()
    # Obtain object point cloud
    obj_pc = gm.UseJSONInput()
    # Calculate grasps and scores
    grasps, scores = gm.GenerateGrasps(point_cloud=obj_pc)
    # Format output
    output= gm.OutputFormatting(grasps=grasps, scores=scores)

    # Dump output and print
    print(json.dumps({"grasps": output}), flush=True)
    print("Number of grasps:")
    print(len(output))

    # Implementation that would retrieve input from a ROS node, not tested
    # if args.server:
    #     gm = GraspGenModule()
    #     line = input()
    #     request = json.loads(line)
    #     obj_pc = gm.RetrieveInput(request)
    #     grasps, scores = gm.GenerateGrasps(obj_pc=obj_pc)
    #     output= gm.OutputFormatting(grasps=grasps, scores=scores)
    #     json.dumps({"grasps": output}), flush=True)


