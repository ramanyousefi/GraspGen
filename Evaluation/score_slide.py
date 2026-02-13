import pybullet as p
import pybullet_data
import numpy as np
import re

# ---------------- CONFIG ----------------
SLIDE_POS = [0.15, 0.15, 0.08]
SLIDE_ORN = [0.850904, 0.0, 0.0, 0.525322]

TXT_FILE = "message2.txt"      # <-- YOUR TXT FILE
URDF_FILE = "slide2.urdf"

F_MAX = 50.0    # normalization constant
alpha = 5.0     # distance decay factor

# ---------------- LOAD GRASPS FROM TXT ----------------
def load_grasps_from_txt(txt_file):
    grasps = []

    with open(txt_file, "r") as f:
        content = f.read()

    blocks = content.split("--------------------")

    for block in blocks:
        if "Position" not in block:
            continue

        pos_match = re.search(r"Position:\s*\[([^\]]+)\]", block)
        rot_match = re.search(r"Rotation:\s*\[([^\]]+)\]", block)
        score_match = re.search(r"Score:\s*([0-9.]+)", block)

        if not (pos_match and rot_match and score_match):
            continue

        pos = [float(x) for x in pos_match.group(1).split(",")]
        rot = [float(x) for x in rot_match.group(1).split(",")]
        score = float(score_match.group(1))

        grasps.append((pos, rot, score))

    return grasps


grasps = load_grasps_from_txt(TXT_FILE)

# ---------------- INIT PYBULLET ----------------
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)
p.loadURDF("plane.urdf")

slide = p.loadURDF(
    URDF_FILE,
    basePosition=SLIDE_POS,
    baseOrientation=SLIDE_ORN,
    useFixedBase=True
)

gripper = p.loadURDF(
    "franka_panda/panda.urdf",
    useFixedBase=True
)

# ---------------- EVAL ----------------
results = []

for i, (gpos, gorn, pred) in enumerate(grasps, 1):
    p.resetBasePositionAndOrientation(slide, SLIDE_POS, SLIDE_ORN)
    p.resetBasePositionAndOrientation(gripper, gpos, gorn)

    for j in [9, 10]:
        p.setJointMotorControl2(
            gripper,
            j,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=100
        )

    for _ in range(240):
        p.stepSimulation()

    contacts = p.getContactPoints(gripper, slide)
    slide_pos, _ = p.getBasePositionAndOrientation(slide)

    # physics score (UNCHANGED)
    physics_score = min(len(contacts) / 10.0, 1.0)

    # epsilon_pb (ADDITIVE)
    epsilon_pb = 0.0
    for c in contacts:
        normal_force = c[9]
        contact_pos = c[6]

        d = np.linalg.norm(
            np.array(contact_pos) - np.array(slide_pos)
        )

        epsilon_pb += normal_force * np.exp(-alpha * d)

    epsilon_pb = min(epsilon_pb / F_MAX, 1.0)

    results.append({
        "id": i,
        "predicted": pred,
        "physics": physics_score,
        "epsilon_pb": epsilon_pb,
        "contacts": len(contacts),
        "z": slide_pos[2]
    })

p.disconnect()

# ---------------- OUTPUT PER GRASP ----------------
for r in results:
    print(
        f"GRASP {r['id']:02d} | "
        f"predicted={r['predicted']:.4f} | "
        f"physics={r['physics']:.4f} | "
        f"epsilon_pb={r['epsilon_pb']:.4f} | "
        f"contacts={r['contacts']} | "
        f"z={r['z']:.4f}"
    )

# ---------------- SUMMARY STATS ----------------
pred_vals = np.array([r["predicted"] for r in results])
phys_vals = np.array([r["physics"] for r in results])
eps_vals  = np.array([r["epsilon_pb"] for r in results])
contact_vals = np.array([r["contacts"] for r in results])

print("\n===== SUMMARY (ALL GRASPS) =====")
print(f"Predicted  | min={pred_vals.min():.4f} | avg={pred_vals.mean():.4f} | max={pred_vals.max():.4f}")
print(f"Physics    | min={phys_vals.min():.4f} | avg={phys_vals.mean():.4f} | max={phys_vals.max():.4f}")
print(f"Epsilon_pb | min={eps_vals.min():.4f} | avg={eps_vals.mean():.4f} | max={eps_vals.max():.4f}")
print(f"Contacts   | min={contact_vals.min():.0f} | avg={contact_vals.mean():.2f} | max={contact_vals.max():.0f}")
