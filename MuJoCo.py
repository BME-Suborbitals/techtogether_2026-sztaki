import mujoco
import mujoco.viewer
import time

model_path = "Unitree_model/g1_dual_arm.xml"
# 1. Modell betöltése
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 2. Szimulációs ablak megnyitása
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 3. Fizikai lépés számítása
        mujoco.mj_step(model, data)
        
        # 4. Megjelenítés frissítése
        viewer.sync()

        # Időzítés tartása
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)