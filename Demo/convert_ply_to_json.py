import numpy as np
import json
import os
import open3d as o3d
import argparse
import sys

# --- HARDCODED OBJECT DEFINITION (Specific to your setup) ---
# These remain hardcoded because they define physical properties of your specific object.
CROP_COORDS = {
    "x_min": -250, "x_max": 260,
    "y_min": -205, "y_max": 140,
    "z_min": -1680, "z_max": -1600
}
OBJECT_HEIGHT = 400 
# ------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert PLY to GraspGen JSON with advanced controls.")
    
    # Required Input
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .ply file")
    
    # Output Control
    parser.add_argument("--output_dir", "-o", type=str, default="graspgen_data", help="Directory to save the JSON")
    
    # Transformation Flags
    parser.add_argument("--center", "-c",default=True, action="store_true", help="If set, moves the object center to (0,0,0)")
    parser.add_argument("--scale", "-s", type=float, default=0.001, help="Scale factor to Meters. Default 0.001 (mm->m). Use 1.0 if already meters.")
    
    # Processing Parameters
    parser.add_argument("--scene_padding", "-p", type=float, default=100.0, help="Padding around object for context (in input units)")
    parser.add_argument("--sample_size", "-n", type=int, default=50000, help="Max points for the object")

    return parser.parse_args()

def process_point_cloud(args):
    # --- 1. Load Data ---
    if not os.path.exists(args.input):
        print(f"❌ Error: File '{args.input}' not found.")
        sys.exit(1)

    print(f"--- Processing: {args.input} ---")
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), [200, 200, 200], dtype=np.uint8)

    # --- 2. Define Bounds (Input Units) ---
    z_floor = CROP_COORDS["z_min"]
    z_ceil = z_floor + OBJECT_HEIGHT
    
    # --- 3. Extract Object ---
    mask_obj = (
        (points[:, 0] >= CROP_COORDS["x_min"]) & (points[:, 0] <= CROP_COORDS["x_max"]) &
        (points[:, 1] >= CROP_COORDS["y_min"]) & (points[:, 1] <= CROP_COORDS["y_max"]) &
        (points[:, 2] >= z_floor)              & (points[:, 2] <= z_ceil)
    )

    obj_pc = points[mask_obj]
    obj_rgb = colors[mask_obj]

    if len(obj_pc) == 0:
        print("🚨 Error: 0 object points found. Check CROP_COORDS.")
        sys.exit(1)

    # Downsample Object
    if len(obj_pc) > args.sample_size:
        idx = np.random.choice(len(obj_pc), args.sample_size, replace=False)
        obj_pc = obj_pc[idx]
        obj_rgb = obj_rgb[idx]

    # --- 4. Extract Scene (Object + Padding) ---
    pad = args.scene_padding
    scene_bounds = {
        "x_min": CROP_COORDS["x_min"] - pad, "x_max": CROP_COORDS["x_max"] + pad,
        "y_min": CROP_COORDS["y_min"] - pad, "y_max": CROP_COORDS["y_max"] + pad,
        "z_min": z_floor - pad,              "z_max": z_ceil + pad
    }

    mask_scene = (
        (points[:, 0] >= scene_bounds["x_min"]) & (points[:, 0] <= scene_bounds["x_max"]) &
        (points[:, 1] >= scene_bounds["y_min"]) & (points[:, 1] <= scene_bounds["y_max"]) &
        (points[:, 2] >= scene_bounds["z_min"]) & (points[:, 2] <= scene_bounds["z_max"])
    )

    scene_pc = points[mask_scene]
    scene_rgb = colors[mask_scene]

    # --- 5. Scale & Transform (The Flags Logic) ---
    print(f"📏 Scaling factor: {args.scale}")
    
    obj_pc_meters = obj_pc * args.scale
    scene_pc_meters = scene_pc * args.scale
    
    if args.center:
        center = np.mean(obj_pc_meters, axis=0)
        print(f"📍 Centering enabled. Shift: {-center.round(3)}")
        obj_pc_meters -= center
        scene_pc_meters -= center
    else:
        print("📍 Centering disabled. Keeping original world coordinates.")

    # --- 6. Recalculate Mask (Post-Transform) ---
    # We map which points in the final scene correspond to the object bounds
    # Note: Logic is scale-invariant, so we can use scaled coords against scaled bounds OR original.
    # We use original mask logic mapped to the subset for safety.
    
    # Re-verify mask on the extracted scene subset
    # (We must use the subset 'scene_pc' before scaling to match 'mask_scene' logic implies)
    # Actually, simpler: Determine mask based on the scaled points against scaled bounds.
    
    scaled_bounds = {k: v * args.scale for k, v in CROP_COORDS.items()}
    # Correction: Bounds are min/max. If scale is negative (unlikely), flip. Assuming positive scale.
    s_z_floor = z_floor * args.scale
    s_z_ceil = z_ceil * args.scale

    # If centered, we must adjust bounds too. But easier: just re-segment based on geometry.
    # Actually, the most robust way is to just use the boolean logic on the final points.
    # However, since we might have shifted, calculating "what is object" is hard if we don't track the shift.
    # Strategy: Compute the mask on the `scene_pc` (original units) BEFORE scaling/shifting.
    
    mask_final_bool = (
        (scene_pc[:, 0] >= CROP_COORDS["x_min"]) & (scene_pc[:, 0] <= CROP_COORDS["x_max"]) &
        (scene_pc[:, 1] >= CROP_COORDS["y_min"]) & (scene_pc[:, 1] <= CROP_COORDS["y_max"]) &
        (scene_pc[:, 2] >= z_floor)              & (scene_pc[:, 2] <= z_ceil)
    )

    # --- 7. Print Detailed Stats ---
    print("-" * 30)
    print("📊 SCENE STATISTICS")
    print(f"   Object Points: {len(obj_pc_meters):,}")
    print(f"   Scene Points:  {len(scene_pc_meters):,}")
    
    # Calculate dimensions in Meters
    obj_min = obj_pc_meters.min(axis=0)
    obj_max = obj_pc_meters.max(axis=0)
    dims = obj_max - obj_min
    
    print(f"   Object Size:   {dims[0]:.3f}m (W) x {dims[1]:.3f}m (D) x {dims[2]:.3f}m (H)")
    print(f"   Object Center: {np.mean(obj_pc_meters, axis=0).round(3)}")
    print("-" * 30)

    # --- 8. Save ---
    data = {
        "object_info": {
            "pc": obj_pc_meters.tolist(),
            "pc_color": obj_rgb.tolist()
        },
        "grasp_info": { "grasp_poses": [], "grasp_conf": [] },
        "scene_info": {
            "pc_color": [scene_pc_meters.tolist()],
            "img_color": scene_rgb.tolist(),
            "img_depth": [],
            "obj_mask": mask_final_bool.astype(int).tolist()
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.input))[0] + ".json"
    out_path = os.path.join(args.output_dir, filename)
    
    with open(out_path, "w") as f:
        json.dump(data, f)
        
    print(f"✅ Output saved: {out_path}")

if __name__ == "__main__":
    args = parse_arguments()
    process_point_cloud(args)