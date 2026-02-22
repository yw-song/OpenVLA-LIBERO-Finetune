import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero.envs import OffScreenRenderEnv
import os
import imageio

# --- 配置 ---
MODEL_PATH = "openvla/openvla-7b"
BDDL_FOLDER = "libero_spatial"
BDDL_FILE = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
DEVICE = "cuda:0"

# 原始模型不包含 'libero_spatial' 的统计量，需使用通用键
UNNORM_KEY = "bridge_orig" 

print("="*50)
print("[1/4] Initializing LIBERO Simulation Environment...")

if "BDDL_FILES" not in os.environ:
    raise ValueError("❌ Error: BDDL_FILES environment variable is missing. Please run 'source ~/.bashrc' first.")

env_args = {
    "bddl_file_name": os.path.join(os.environ["BDDL_FILES"], BDDL_FOLDER, BDDL_FILE),
    "camera_heights": 256,
    "camera_widths": 256,
    "camera_depths": False,
}

try:
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    print(f"✅ Simulation Ready. Loaded: {BDDL_FILE}")
except Exception as e:
    print(f"❌ Failed to load environment: {e}")
    exit()

print("\n[2/4] Loading OpenVLA Model (Flash Attention)...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(DEVICE)
print("✅ Model Loaded.")

print("\n[3/4] Starting Robot Inference Loop (30 steps demo)...")

# 提示词格式
prompt = "In: What action should the robot take to {}?\nOut:"
instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"

frames = [] 
obs = env.reset()

for step in range(30): 
    img_array = obs["agentview_image"] 
    image = Image.fromarray(img_array[::-1]) 
    frames.append(img_array[::-1]) 

    inputs = processor(text=prompt.format(instruction), images=image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

    obs, reward, done, info = env.step(action)
    print(f"\rStep {step+1}/30: Action Generated -> Executed", end="", flush=True)

print("\n\n[4/4] Saving Video...")
video_path = "robot_demo.mp4"
imageio.mimsave(video_path, np.stack(frames), fps=10)
print(f"🎉🎉🎉 Video saved to: {os.path.abspath(video_path)}")
print("="*50)