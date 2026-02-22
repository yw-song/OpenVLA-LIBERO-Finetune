import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# --- 1. 基础环境检查 ---
print("="*50)
print("[Step 1] Checking Hardware & Drivers...")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device_name = torch.cuda.get_device_name(0)
print(f"GPU Device: {device_name}")

# 检查 Flash Attention 是否可用 (A800 必备)
try:
    # 尝试导入 flash_attn 包
    import flash_attn
    print("Flash Attention 2 Package: INSTALLED ✅")
    use_flash = True
except ImportError:
    print("Flash Attention 2 Package: NOT FOUND ❌ (Running slow mode)")
    use_flash = False

# --- 2. 仿真环境检查 ---
print("\n[Step 2] Checking Simulation Environment...")
# 强制使用 EGL 后端 (因为服务器没有显示器)
os.environ['MUJOCO_GL'] = 'egl' 
try:
    import mujoco
    import libero.libero
    print("MuJoCo & LIBERO: IMPORT SUCCESS ✅")
except Exception as e:
    print(f"LIBERO Error: {e} ❌")

# --- 3. 模型加载测试 ---
print("\n[Step 3] Loading OpenVLA-7B (Downloading ~15GB)...")
print("⚠️  This may take 5-10 minutes. Please wait...")

model_id = "openvla/openvla-7b"

try:
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # 加载模型 (自动使用 Flash Attention 2)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        attn_implementation="flash_attention_2" if use_flash else "eager",
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    
    print("\n🎉🎉🎉 SUCCESS! OpenVLA loaded on " + device_name)
    print("Environment is 100% READY for experiments.")

except Exception as e:
    print(f"\n❌ Model Load Failed: {e}")
    print("Possible reasons: HF Token not valid, Network issue, or Flash Attn mismatch.")

print("="*50)