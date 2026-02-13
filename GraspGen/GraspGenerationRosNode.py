#!/usr/bin/env python3

import rospy
import json
import subprocess
import threading
import numpy as np

CONDA_ENV_NAME = "graspgen"
PYTHON_EXEC = "python"
GRASPGEN_MODULE_PATH = "graspgen_script.py" # Check if pathing is correct

INPUT_TOPIC = "/input_source"
OUTPUT_TOPIC = "/grasp_generation"

# This ROS node was created with the intention of being used to integrate the grasp generation process with the larger project
# However, due to integration input, it has never been connected or tested to work with other nodes
# It can still serve as a starting point for future development

class GraspGenProcess:
    def __init__(self):
        # Launch the actual grasp generation script in the correct environment through a subprocess
        self.proc = subprocess.Popen(
            [
                "conda", "run", "-n", CONDA_ENV_NAME,
                PYTHON_EXEC, GRASPGEN_MODULE_PATH,
                "--server"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Create a thread lock for later
        self.lock = threading.Lock()

    def GenerateGrasps(self, pointcloud, obj_mask):
        # Create request based on inputs
        request = {
            "pointcloud": pointcloud,
            "objectmask": obj_mask,
            "frame": "world",
            "path": "path/to/json.json"
        }

        # Prevent multiple threads from talking to the subprocess
        with self.lock:
            # Send request to the script
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()

            # Receive response from script
            response = self.proc.stdout.readline()
            result = json.loads(response)

        # Return grasps
        return result["grasps"]

class GraspGenROSNode:

    # Initiate subscriber and publisher node
    def __init__(self):
        rospy.init_node("graspgen_node")

        # GraspGen object needed for generating grasps
        self.graspgen = GraspGenProcess()

        # Create publisher node
        self.pub = rospy.Publisher(
            OUTPUT_TOPIC,
            queue_size=1
        )

        # Create subscriber node
        self.sub = rospy.Subscriber(
            INPUT_TOPIC,
            self.callback,
            queue_size=1
        )

        rospy.loginfo("Grasp generation node ready")

    # Callback function called when subscribed node receives necessary input
    def callback(self):
        rospy.loginfo("Grasp generation node callback function activated")

        # Some placeholders for actual data
        pc = "pointcloud"
        obj_mask = "object mask"

        # Generate grasps
        grasps = self.graspgen.generate_grasps(pc, obj_mask)

        # Publish generated grasps
        self.pub.publish(grasps)
        rospy.loginfo(f"Published {len(grasps)} grasps")

# Start
if __name__ == "__main__":
    try:
        node = GraspGenROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
