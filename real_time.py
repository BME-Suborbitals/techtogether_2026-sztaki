"""
Real-time MediaPipe Pose -> Unitree G1 MuJoCo retargeting using mink IK.
Now uses MediaPipe WORLD landmarks for true 3D wrist tracking, so the
elbow can actually extend and retract.
"""

import time
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer

import mink

import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

# ============================================================
# CONFIG
# ============================================================
G1_XML_PATH = "unitree-g1-mujoco/assets/g1_dual_arm.xml"

ARM_LEFT  = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
             "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
ARM_RIGHT = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint",
             "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

LEFT_HAND_FRAME  = "left_wrist_yaw_link"
RIGHT_HAND_FRAME = "right_wrist_yaw_link"
LEFT_SHOULDER_BODY  = "left_shoulder_pitch_link"
RIGHT_SHOULDER_BODY = "right_shoulder_pitch_link"

KP_ARM, KD_ARM     = 150.0, 8.0
KP_WRIST, KD_WRIST = 30.0, 1.5

# MOD: Removed HUMAN_ARM_LEN_MP and FORWARD_OFFSET — no longer needed.
# Scale is now computed per-frame from world landmarks (which are in meters).

# ============================================================
# MEDIAPIPE
# ============================================================
pose = mp_pose.Pose(
    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)

# ============================================================
# MUJOCO
# ============================================================
model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

model.opt.gravity[:] = [0, 0, 0]

if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

# Sensible rest pose: arms hanging down with slight elbow bend
def set_qpos(name, value):
    try:    data.qpos[model.joint(name).qposadr[0]] = value
    except KeyError: pass

set_qpos("left_shoulder_pitch_joint",  0.0)
set_qpos("left_shoulder_roll_joint",   0.0)
set_qpos("left_shoulder_yaw_joint",    0.0)
set_qpos("left_elbow_joint",           0.1)
set_qpos("right_shoulder_pitch_joint", 0.0)
set_qpos("right_shoulder_roll_joint",  0.0)
set_qpos("right_shoulder_yaw_joint",   0.0)
set_qpos("right_elbow_joint",          0.1)

mujoco.mj_forward(model, data)

def body_pos(name):
    return data.body(name).xpos.copy()

robot_l_arm_len = np.linalg.norm(body_pos(LEFT_HAND_FRAME)  - body_pos(LEFT_SHOULDER_BODY))
robot_r_arm_len = np.linalg.norm(body_pos(RIGHT_HAND_FRAME) - body_pos(RIGHT_SHOULDER_BODY))
robot_arm_len = (robot_l_arm_len + robot_r_arm_len) / 2.0
max_reach = robot_arm_len * 0.95
print(f"[INFO] Robot arm length: {robot_arm_len:.3f} m  Max reach: {max_reach:.3f} m")

def build_arm_table(names):
    table = []
    for n in names:
        try:
            aid = model.actuator(n).id
            qa  = int(model.joint(n).qposadr[0])
            qva = int(model.joint(n).dofadr[0])
        except KeyError:
            print(f"[WARN] '{n}' not found"); continue
        is_wrist = "wrist" in n
        kp = KP_WRIST if is_wrist else KP_ARM
        kd = KD_WRIST if is_wrist else KD_ARM
        cmin, cmax = model.actuator(aid).ctrlrange
        if cmin == 0.0 and cmax == 0.0:
            print(f"[ERROR] '{n}' has ctrlrange=[0, 0] — fix MJCF!")
        table.append((aid, qa, qva, kp, kd, cmin, cmax))
    return table

L_ARM = build_arm_table(ARM_LEFT)
R_ARM = build_arm_table(ARM_RIGHT)
print(f"[INFO] Resolved {len(L_ARM)} left, {len(R_ARM)} right arm joints")

# ============================================================
# MINK
# ============================================================
configuration = mink.Configuration(model)
configuration.update(data.qpos)

left_hand_task = mink.FrameTask(
    frame_name=LEFT_HAND_FRAME, frame_type="body",
    position_cost=1.0, orientation_cost=0.0,
)
right_hand_task = mink.FrameTask(
    frame_name=RIGHT_HAND_FRAME, frame_type="body",
    position_cost=1.0, orientation_cost=0.0,
)
posture_task = mink.PostureTask(model, cost=1e-5)
posture_task.set_target_from_configuration(configuration)

tasks  = [left_hand_task, right_hand_task, posture_task]
limits = [mink.ConfigurationLimit(model)]

# ============================================================
# RETARGETING — uses WORLD landmarks (3D meters, hip-centered)
# ============================================================
def clamp_to_shoulder(target, shoulder):
    delta = target - shoulder
    d = np.linalg.norm(delta)
    if d > max_reach:
        target = shoulder + delta * (max_reach / d)
    return target

def mp_wrist_targets(world_lm):
    """Convert MediaPipe WORLD landmarks (meters, hip-centered) to robot frame.

    MediaPipe world frame:  x=person's right, y=down,    z=behind person
    Robot (G1) frame:       x=forward,        y=left,    z=up
    So:  robot_x = -mp_z   (forward = away from camera = -mp_z, which is +behind)
         robot_y = -mp_x   (left in robot = opposite of person's right)
         robot_z = -mp_y   (up in robot = opposite of down in MP)
    """
    L = mp_pose.PoseLandmark
    def p(i): return np.array([world_lm[i].x, world_lm[i].y, world_lm[i].z])

    # MOD: Wrist position relative to shoulder, ALREADY IN METERS thanks to
    # world_landmarks. No scaling needed for the user-relative offset.
    l_rel_user = p(L.LEFT_WRIST.value)  - p(L.LEFT_SHOULDER.value)
    r_rel_user = p(L.RIGHT_WRIST.value) - p(L.RIGHT_SHOULDER.value)

    # MOD: Scale user's arm length to robot's arm length.
    # Compute user's actual arm length from world landmarks each frame.
    user_l_arm = (np.linalg.norm(p(L.LEFT_ELBOW.value)   - p(L.LEFT_SHOULDER.value)) +
                  np.linalg.norm(p(L.LEFT_WRIST.value)   - p(L.LEFT_ELBOW.value)))
    user_r_arm = (np.linalg.norm(p(L.RIGHT_ELBOW.value)  - p(L.RIGHT_SHOULDER.value)) +
                  np.linalg.norm(p(L.RIGHT_WRIST.value)  - p(L.RIGHT_ELBOW.value)))
    user_arm   = (user_l_arm + user_r_arm) / 2.0

    # Guard against degenerate values (e.g. landmarks not yet detected)
    if user_arm < 0.1:
        user_arm = 0.5
    scale = robot_arm_len / user_arm

    l_rel_user *= scale
    r_rel_user *= scale

    # MOD: Coordinate transform MediaPipe world -> G1 robot frame
    def to_robot(v):
        return np.array([-v[2], -v[0], -v[1]])

    l_target = body_pos(LEFT_SHOULDER_BODY)  + to_robot(l_rel_user)
    r_target = body_pos(RIGHT_SHOULDER_BODY) + to_robot(r_rel_user)

    l_target = clamp_to_shoulder(l_target, body_pos(LEFT_SHOULDER_BODY))
    r_target = clamp_to_shoulder(r_target, body_pos(RIGHT_SHOULDER_BODY))
    return l_target, r_target, scale

# ============================================================
# PD CONTROLLER
# ============================================================
def apply_pd(arm_table, q_targets, qdot_targets):
    for (aid, qa, qva, kp, kd, cmin, cmax), q_t, qd_t in zip(arm_table, q_targets, qdot_targets):
        q     = data.qpos[qa]
        qdot  = data.qvel[qva]
        tau   = kp * (q_t - q) + kd * (qd_t - qdot)
        data.ctrl[aid] = np.clip(tau, cmin, cmax)

# ============================================================
# STATE
# ============================================================
state = {
    "mode": 1, "paused": False, "sweep_t": 0.0,
    "current_scale": 1.0,
    "l_q_target":    np.array([data.qpos[model.joint(n).qposadr[0]] for n in ARM_LEFT]),
    "l_qdot_target": np.zeros(len(L_ARM)),
    "r_q_target":    np.array([data.qpos[model.joint(n).qposadr[0]] for n in ARM_RIGHT]),
    "r_qdot_target": np.zeros(len(R_ARM)),
}

def key_callback(keycode):
    try: ch = chr(keycode)
    except ValueError: return
    if ch in ("1", "2", "3"):
        state["mode"] = int(ch); state["sweep_t"] = 0.0
        print(f"[KEY] Mode -> {ch}")
    elif ch == " ":
        state["paused"] = not state["paused"]
        print(f"[KEY] Paused: {state['paused']}")

# ============================================================
# MAIN LOOP
# ============================================================
cam = cv.VideoCapture(0)
dt    = model.opt.timestep
ik_dt = 1.0 / 30.0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while cam.isOpened() and viewer.is_running():
        success, frame = cam.read()
        if not success:
            print("Camera Frame not available"); continue
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pose_detected = pose.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if state["mode"] == 1:
            left_hand_task.set_target(mink.SE3.from_translation(body_pos(LEFT_HAND_FRAME)))
            right_hand_task.set_target(mink.SE3.from_translation(body_pos(RIGHT_HAND_FRAME)))

        elif state["mode"] == 2:
            # MOD: Sweep now also varies x (depth) so the elbow has to extend/retract
            state["sweep_t"] += ik_dt
            t = state["sweep_t"]
            center = body_pos(LEFT_SHOULDER_BODY) + np.array([0.25, 0.0, -0.05])
            target = center + np.array([
                0.15 * np.sin(t * 0.7),   # depth oscillation -> elbow extends/retracts
                0.15 * np.sin(t),
                0.15 * np.cos(t),
            ])
            left_hand_task.set_target(mink.SE3.from_translation(target))
            right_hand_task.set_target(mink.SE3.from_translation(body_pos(RIGHT_HAND_FRAME)))

        elif state["mode"] == 3 and pose_detected.pose_world_landmarks:
            # MOD: Now use pose_WORLD_landmarks (3D in meters, not image-normalized)
            l_target, r_target, scl = mp_wrist_targets(pose_detected.pose_world_landmarks.landmark)
            state["current_scale"] = scl
            left_hand_task.set_target(mink.SE3.from_translation(l_target))
            right_hand_task.set_target(mink.SE3.from_translation(r_target))

        if not state["paused"]:
            configuration.update(data.qpos)
            velocity = mink.solve_ik(configuration, tasks, ik_dt,
                                     solver="quadprog", limits=limits)
            new_qpos = configuration.integrate(velocity, ik_dt)

            for i, (aid, qa, qva, *_) in enumerate(L_ARM):
                state["l_q_target"][i]    = new_qpos[qa]
                state["l_qdot_target"][i] = velocity[qva]
            for i, (aid, qa, qva, *_) in enumerate(R_ARM):
                state["r_q_target"][i]    = new_qpos[qa]
                state["r_qdot_target"][i] = velocity[qva]

            n_substeps = max(1, int(ik_dt / dt))
            for _ in range(n_substeps):
                apply_pd(L_ARM, state["l_q_target"], state["l_qdot_target"])
                apply_pd(R_ARM, state["r_q_target"], state["r_qdot_target"])
                mujoco.mj_step(model, data)

        viewer.sync()

        if pose_detected.pose_landmarks:
            drawing.draw_landmarks(
                frame, pose_detected.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            )
        hud1 = f"Mode {state['mode']}  {'PAUSED' if state['paused'] else 'RUN'}  [1/2/3/SPACE/q]"
        hud2 = f"robot_arm={robot_arm_len:.2f}m  scale={state['current_scale']:.2f}"
        cv.putText(frame, hud1, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(frame, hud2, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv.imshow("Pose -> G1 (mink IK)", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv.destroyAllWindows()
