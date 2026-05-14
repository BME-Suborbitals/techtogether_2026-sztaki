"""
Diagnostic: dump everything the G1 model exposes.
No camera, no mediapipe, no viewer - just inspect the model.
"""
import mujoco

G1_XML_PATH = "unitree-g1-mujoco/assets/g1_dual_arm.xml"

model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
data  = mujoco.MjData(model)

print(f"\n=== MODEL SUMMARY ===")
print(f"nq (position DOFs):  {model.nq}")
print(f"nv (velocity DOFs):  {model.nv}")
print(f"nu (actuators):      {model.nu}")
print(f"njnt (joints):       {model.njnt}")
print(f"nbody (bodies):      {model.nbody}")
print(f"nkey (keyframes):    {model.nkey}")
print(f"timestep:            {model.opt.timestep}")

print(f"\n=== ALL JOINTS ===")
for i in range(model.njnt):
    j = model.joint(i)
    jtype = ["FREE", "BALL", "SLIDE", "HINGE"][j.type[0]]
    print(f"  [{i:2d}] {j.name:40s} type={jtype}")

print(f"\n=== ALL ACTUATORS ===")
for i in range(model.nu):
    a = model.actuator(i)
    # ctrlrange shows the valid range of ctrl values for this actuator
    print(f"  [{i:2d}] {a.name:40s} ctrlrange={a.ctrlrange}")

print(f"\n=== ARM-RELATED ACTUATORS (search) ===")
for i in range(model.nu):
    name = model.actuator(i).name
    if "shoulder" in name or "elbow" in name or "wrist" in name:
        print(f"  [{i:2d}] {name}")