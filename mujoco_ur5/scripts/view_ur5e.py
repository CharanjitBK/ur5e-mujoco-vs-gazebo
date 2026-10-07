import mujoco
import mujoco.viewer

MODEL_PATH = "/root/ur5_project/mujoco_ur5/models/ur5e_visual_test.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
