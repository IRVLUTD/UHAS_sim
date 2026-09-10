# Adding a New Dexterous Hand to the Simulation

This guide explains how to add a completely new dexterous hand to the UHAS simulation framework. It covers two main parts:

1. **Importing your hand into NVIDIA Isaac Sim** (converting URDF → USD and configuring it properly).
2. **Creating the Unified Hand Action Space (UHAS)** for your hand so it can be controlled with our unified sphere-based policy.

Once completed, your hand can be used with both the baseline joint controller and our full UHAS multi-hand policies (including zero-shot deployment).

<div align="center">
  <img src="../docs/illustrations/sphere_creation.jpg" alt="Overview of adding a new hand: URDF → USD → Sphere creation → UHAS integration" width="75%">
</div>

**Quick note on CIK**: The Cascade Inverse Kinematics (CIK) algorithm is what makes UHAS work across different hands. It classifies each joint as either "lateral" (side-to-side motion) or "encompassing" (conforming to the sphere surface) and solves them in a fast cascaded manner. This allows a single policy to output sphere deformations that are automatically converted into valid joint commands for *your* hand.

---

## Table of Contents

- [Adding a New Dexterous Hand to the Simulation](#adding-a-new-dexterous-hand-to-the-simulation)
  - [Table of Contents](#table-of-contents)
  - [Importing a URDF into Isaac Sim](#importing-a-urdf-into-isaac-sim)
    - [Step-by-step: Convert URDF to USD](#step-by-step-convert-urdf-to-usd)
    - [Verify the Hand Works](#verify-the-hand-works)
  - [Creating the Python Configuration File](#creating-the-python-configuration-file)
  - [Creating the UHAS for a New Hand](#creating-the-uhas-for-a-new-hand)
    - [Required Frames in Your URDF](#required-frames-in-your-urdf)
    - [Open-Hand Configuration (`config.json`)](#open-hand-configuration-configjson)
  - [Processing the URDF with `process_urdf.py`](#processing-the-urdf-with-process_urdfpy)
    - [Important Caveats](#important-caveats)
    - [Script Arguments](#script-arguments)
    - [Running the Script](#running-the-script)
  - [Wiring the Hand into the Simulation](#wiring-the-hand-into-the-simulation)
    - [Baseline Simulation](#baseline-simulation)
    - [UHAS Simulation](#uhas-simulation)
  - [Validation \& Debugging](#validation--debugging)

---

## Importing a URDF into Isaac Sim

We will use the [**Wuji Hand**](https://github.com/wuji-technology/wuji-description) as a running example. The process is the same for any new hand.

### Step-by-step: Convert URDF to USD

1. Open **Isaac Sim 4.5.0**.
2. Search for your `.urdf` file in the content tab.
3. Import your hand's `.urdf` file by double clicking your file.
4. In the import options:
   - Set **Collider Approximation** to **Convex Hull** (recommended for dexterous hands — gives stable and reasonably accurate collision geometry).
   - Enable **Self Collision** if your hand requires it.
   - Enable **Allow Self-Collision**.

5. Click **Convert**.

After import, you should see your hand in the stage. 

**Recommended folder structure** (follow the existing grippers in the repo):

```
grippers/
└── wuji/
    └── wuji_right/
        ├── usd/..         # The converted USD folder
        ├── wuji_right.urdf         # Original URDF (for reference)
        ├── meshes/                 # All .stl / .obj files
        └── config.json             # Open-hand configuration (created later)
```

### Verify the Hand Works

- Play the simulation and manually move the joints using the **Joint Inspector** or ** articulation** controls.
- Check that the fingers move without exploding or penetrating themselves.
- If you see unstable collisions, go to the collider properties and add **Filtered Collider Pairs** between problematic links (common between adjacent finger phalanges).

<div align="center">
  <img src="../docs/illustrations/wuji_loaded.png" alt="Wuji Hand successfully loaded in Isaac Sim with Convex Hull colliders" width="70%">
  <p><em>Wuji Hand loaded in Isaac Sim (Convex Hull colliders)</em></p>
</div>

---

## Creating the Python Configuration File

Create a Python file that defines your hand as an `ArticulationCfg`. This file tells Isaac Lab how to spawn and control your hand.

**Example**: `/grippers/wuji/wuji_right/wuji_right.py`

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "wuji_right", "wuji_right.usd"))

WUJI_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Finger 1 (thumb)
            "right_finger1_joint1": 0.048,
            "right_finger1_joint2": 0.0,
            "right_finger1_joint3": 0.0,
            "right_finger1_joint4": 0.0,
            # Finger 2
            "right_finger2_joint1": 0.0,
            "right_finger2_joint2": 0.0,
            "right_finger2_joint3": 0.0,
            "right_finger2_joint4": 0.0,
            # Finger 3
            "right_finger3_joint1": 0.0,
            "right_finger3_joint2": 0.0,
            "right_finger3_joint3": 0.0,
            "right_finger3_joint4": 0.0,
            # Finger 4
            "right_finger4_joint1": 0.0,
            "right_finger4_joint2": 0.0,
            "right_finger4_joint3": 0.0,
            "right_finger4_joint4": 0.0,
            # Finger 5
            "right_finger5_joint1": 0.0,
            "right_finger5_joint2": 0.0,
            "right_finger5_joint3": 0.0,
            "right_finger5_joint4": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[1-5]_joint[1-4]"],
            effort_limit_sim={"right_finger.*_joint.*": 0.4},
            stiffness={"right_finger.*_joint.*": 3.0},
            damping={"right_finger.*_joint.*": 0.2},
            velocity_limit_sim={"right_finger.*_joint.*": 5.0},
            friction=0.02,
            armature=0.00149376,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

Place this file next to your USD folder following the existing gripper structure.

---

## Creating the UHAS for a New Hand

The UHAS creation process builds a canonical sphere, establishes dense surface correspondences between the sphere and your hand, and generates the Cascade Inverse Kinematics (CIK) mapping.

<div align="center">
  <img src="../docs/illustrations/wuji_mesh_load.png" alt="UHAS creation pipeline: URDF → Sphere → Surface correspondences → CIK" width="600">
</div>

### Required Frames in Your URDF

Before running the processing script, you must add two important frames to your URDF (or USD):

- **Palm frame** (`palm_link` or similar): Its **+Z axis must point outward along the palm normal** (direction the fingers close toward).
- **Fingertip frames** (one per finger): Their **+Z axis must point along the fingertip normal**.

**Recommendation**: Use the **Isaac Sim GUI** to create and align these frames visually — it is much easier than editing the URDF by hand.

<div align="center">
  <img src="../docs/illustrations/wuji_palm_fts.png" alt="Correct palm and fingertip frame orientations for UHAS" width="600">
</div>

### Open-Hand Configuration (`config.json`)

Create a `config.json` file inside your hand folder with the joint positions that correspond to a fully **open** hand:

```json
{
    "opened_dofs": {
        "right_finger1_joint1": 0.0475,
        "right_finger1_joint2": 0.0,
        "right_finger1_joint3": 0.0,
        "right_finger1_joint4": 0.0,
        "right_finger2_joint1": 0.0,
        "right_finger2_joint2": 0.0,
        "right_finger2_joint3": 0.0,
        "right_finger2_joint4": 0.0,
        "right_finger3_joint1": 0.0,
        "right_finger3_joint2": 0.0,
        "right_finger3_joint3": 0.0,
        "right_finger3_joint4": 0.0,
        "right_finger4_joint1": 0.0,
        "right_finger4_joint2": 0.0,
        "right_finger4_joint3": 0.0,
        "right_finger4_joint4": 0.0,
        "right_finger5_joint1": 0.0,
        "right_finger5_joint2": 0.0,
        "right_finger5_joint3": 0.0,
        "right_finger5_joint4": 0.0
    }
}
```

This file is used by the processing script to know the reference "open" pose.

---

## Processing the URDF with `process_urdf.py`

The script that creates the UHAS representation is located at:

```bash
process_urdf/process_urdf.py
```

### Important Caveats

- Some URDFs (especially those exported from MuJoCo) contain extra tags or attributes that the parser doesn't like. You may need to **remove MuJoCo-specific elements** (e.g., `<mujoco>`, custom inertia tags, etc.) before running.
- Make sure all mesh paths in the URDF are relative and correct.

### Script Arguments

```python
def make_parser():
    parser = argparse.ArgumentParser(description='Process urdf and create the sphere controller.')
    parser.add_argument('--robot_path', type=str, help='Path to urdf')
    parser.add_argument('--base_link', type=str, default=None,
                        help='Father link of all fingers in robot.urdf. If omitted, the URDF root is detected.')
    parser.add_argument('--thumb_anchor', type=float, help='Theta Position to place the thumb in', default=1.571)
    parser.add_argument('--verbose', type=bool, help='Running Program in verbose mode',
                        default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--correct_axes', type=bool, help='Correct the joint positions to center to links, use when the joint origins are not centered in the joint child mesh',  default=True, action = argparse.BooleanOptionalAction)
    return parser
```

**Key parameters**:
- `--robot_path`: Path to your `.urdf` file.
- `--base_link`: Palm / father link of all fingers. Optional — if omitted, the URDF kinematic root is detected automatically.
- `--thumb_anchor`: Azimuthal angle (in radians) where the thumb should be placed on the sphere (default ≈ π/2).
- `--verbose`: **Strongly recommended** on first runs. It will display many intermediate images so you can visually verify each step.
- `--correct_axes`: Helps fix joint axis centering in URDFs (on by default). Pass `--no-correct_axes` to disable.

### Running the Script

```bash
cd process_urdf
python process_urdf.py --robot_path ./grippers/wuji/wuji_right/wuji_right.urdf \
                       --thumb_anchor 1.571 \
                       --verbose
```

When running in verbose mode, the script will show the step-by-step construction of the sphere and surface correspondences.

At the very end, it displays **5 sample sphere deformations** so you can visually validate that the CIK mapping works correctly for your hand.

<div align="center">
  <img src="../docs/illustrations/wuji_uhas.png" alt="Intermediate steps shown in verbose mode" width="600">
  <p><em>UHAS correspondences with WUJI Hand</em></p>
</div>

<div align="center">
  <img src="../docs/illustrations/wuji_sample.png" alt="Final 5 sample sphere deformations for validation" width="600">
  <p><em>Final validation: Sample sphere deformation generated for your hand</em></p>
</div>

After successful execution, the script will create a **`sphere_cik.json`** file inside your hand's URDF directory. This file contains everything needed for real-time CIK control.

---

## Wiring the Hand into the Simulation

### Baseline Simulation

Edit your environment configuration file and add:

```python
from sphere_ctrl_isaaclab.assets.grippers.wuji.wuji_right.wuji_right import WUJI_HAND_CFG as WUJI_HAND_RIGHT_CFG

# Inside the robot selection if-statement:
elif robot == "wuji_right":
    robot_cfg: ArticulationCfg = WUJI_HAND_RIGHT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "wuji", robot, "sphere_cik.json")
    action_space = 20   # 5 fingers × 4 joints (adjust if your hand has different DOF)
```

### UHAS Simulation

Edit the UHAS environment configuration:

```python
from sphere_ctrl_isaaclab.assets.grippers.wuji.wuji_right.wuji_right import WUJI_HAND_CFG as WUJI_HAND_RIGHT_CFG

# Inside the hand selection if-statement:
elif robot == "wuji_right":
    robot_cfg: ArticulationCfg = WUJI_HAND_RIGHT_CFG.replace(
        prim_path=robot_path, 
        actuators={"fingers": articulation_cfg}
    )
    cik_json_path = os.path.join(IRVL_ASSET_PATH, "grippers", "wuji", robot, "sphere_cik.json")
```

---

## Validation & Debugging

After wiring the hand, validate it works:

1. Run the `random_agent.py` script with your new hand to check basic control.

<div align="center">
  <img src="../docs/gif/wuji_rand.gif" alt="Wuji hand baseline moving with random agent" width="600">
  <p><em>Wuji hand being controlled random agent actions</em></p>
</div>


1. For UHAS-specific debugging, run the UHAS simulation wwith the `debug_fingers` flag inside the multi_env_cfg.py file.


This mode visualizes how each finger is being driven by the sphere deformations in real time — extremely useful for verifying that CIK is working correctly on your new hand.

<div align="center">
  <img src="../docs/gif/debug_fingers.gif" alt="Wuji hand moving under UHAS control with debug_fingers=True" width="800">
  <p><em>Hands being controlled through UHAS with debug visualization enabled</em></p>
</div>


---

Congratulations! Your new hand is now fully integrated into both the baseline and UHAS simulation pipelines. You can now train policies on it, run zero-shot transfer from other hands, or fine-tune existing multi-hand policies.

If you run into any issues during the URDF conversion or sphere creation process, feel free to open an issue with the verbose output images — they are usually very helpful for debugging.
