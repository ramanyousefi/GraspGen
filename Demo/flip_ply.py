import open3d as o3d
import numpy as np
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Flip a specific axis of a PLY point cloud.")
    
    # Required Input
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .ply file")
    
    # Output (Optional)
    parser.add_argument("--output", "-o", type=str, default="fixed.ply", help="Path to save output file (default: fixed.ply)")
    
    # Axis Selection (Default is Z)
    parser.add_argument("--axis", "-a", type=str, default="z", choices=["x", "y", "z"], 
                        help="Which axis to invert? (x, y, or z). Default is z.")

    return parser.parse_args()

def flip_cloud(args):
    # 1. Check File
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"--- Loading {args.input} ---")
    pcd = o3d.io.read_point_cloud(args.input)
    
    if pcd.is_empty():
        print("❌ Error: Point cloud is empty or invalid format.")
        sys.exit(1)

    # 2. Convert to Numpy
    pts = np.asarray(pcd.points)
    print(f"   Points: {len(pts)}")
    print(f"   Original Center: {pts.mean(axis=0).round(3)}")

    # 3. Flip Axis
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map[args.axis.lower()]
    
    print(f"🔄 Flipping {args.axis.upper()} axis...")
    pts[:, idx] *= -1

    # 4. Update and Save
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"   New Center:      {pts.mean(axis=0).round(3)}")
    
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"✅ Saved to: {args.output}")

if __name__ == "__main__":
    args = parse_args()
    flip_cloud(args)