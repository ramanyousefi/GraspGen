import subprocess
import argparse
import os
import sys
import time

# --- CONFIGURATION ---
# Default file if no argument is given
DEFAULT_INPUT = "1-slider-in-box.ply"

# Paths (Relative to this script)
GRIPPER_CONFIG = "/models/checkpoints/graspgen_robotiq_2f_140.yml"
OUTPUT_DIR = "graspgen_data"
# ---------------------

# ANSI Colors for professional terminal output
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_step(command, step_name):
    print(f"{CYAN}[Pipeline] Executing: {step_name}...{RESET}")
    # print(f"  $ {command}") # Uncomment to debug
    try:
        subprocess.check_call(command, shell=True)
        print(f"{GREEN}✔ {step_name} Complete.{RESET}\n")
    except subprocess.CalledProcessError:
        print(f"\n{YELLOW}❌ Critical Error during: {step_name}{RESET}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Zivid Vision -> GraspGen ROS Pipeline")
    
    # This makes it "look like you can give different file"
    parser.add_argument("input_file", nargs="?", default=DEFAULT_INPUT, 
                        help="Path to the input pointcloud from camera (e.g. scan.ply)")
    
    args = parser.parse_args()

    # Verify Input
    if not os.path.exists(args.input_file):
        print(f"{YELLOW}Error: Input file '{args.input_file}' not found.{RESET}")
        sys.exit(1)

    print(f"{GREEN}================================================={RESET}")
    print(f"{GREEN}   STARTING GRASP GENERATION PIPELINE   {RESET}")
    print(f"{GREEN}   Input: {args.input_file}   {RESET}")
    print(f"{GREEN}================================================={RESET}\n")

    # ---------------------------------------------------------
    # STEP 1: FLIP / NORMALIZE (flip_ply.py)
    # ---------------------------------------------------------
    # Creates 'fixed.ply'
    cmd_flip = f"python3 flip_ply.py --input {args.input_file} --output fixed.ply --axis z"
    run_step(cmd_flip, "Normalizing Point Cloud (Z-Flip)")

    # ---------------------------------------------------------
    # STEP 2: CONVERT TO JSON (convert_ply_to_json.py)
    # ---------------------------------------------------------
    # Creates 'graspgen_data/fixed.json'
    # We do NOT center it because you wanted original coordinates
    cmd_convert = (
        f"python3 convert_ply_to_json.py "
        f"--input fixed.ply "
        f"--output_dir {OUTPUT_DIR} "
        f"--scale 0.001 " # Ensuring MM -> M conversion
    )
    run_step(cmd_convert, "Generating Inference JSON")

    # ---------------------------------------------------------
    # STEP 3: RUN GRASPGEN (my_scene_pc.py)
    # ---------------------------------------------------------
    # Note: We use 2000 grasps for high density, filter collisions enabled
    cmd_inference = (
        f"python3 my_scene_pc.py "
        f"--sample_data_dir {OUTPUT_DIR} "
        f"--gripper_config {GRIPPER_CONFIG} "
        f"--num_grasps 2000 "
        f"--filter_collisions"
    )
    run_step(cmd_inference, "Running GraspGen Inference")

    # ---------------------------------------------------------
    # FINISH
    # ---------------------------------------------------------
    expected_output = os.path.join(OUTPUT_DIR, "exported_grasps", "fixed_robot_grasps.json")
    
    print(f"{GREEN}================================================={RESET}")
    print(f"{GREEN}   PIPELINE SUCCESSFUL {RESET}")
    if os.path.exists(expected_output):
        print(f"{CYAN}   Grasps exported to: {expected_output}{RESET}")
        print(f"{CYAN}   Ready for ROS Node transmission.{RESET}")
    else:
        print(f"{YELLOW}   Warning: GraspGen finished, but output file not found at expected path.{RESET}")
    print(f"{GREEN}================================================={RESET}")

if __name__ == "__main__":
    main()