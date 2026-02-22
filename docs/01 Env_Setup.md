# OpenVLA 环境配置

## 1. 初始化系统与 Conda

由于 NGC 镜像默认不带 Conda，且系统盘较小，需在数据盘手动安装。

```bash
# 1. 进入数据盘
cd /datadisk

# 2. 安装系统级渲染依赖 (LIBERO/MuJoCo 必须)
apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf git vim

# 3. 安装 Miniconda
# 可以用国内镜像源代替，但现在速度可接受
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /datadisk/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# 4. 初始化环境变量
/datadisk/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 2. 解决协议与创建环境

解决 Anaconda 的 ToS 协议报错，并建立隔离环境

```bash
# 1. 接受协议 (防止报错)
conda config --set channel_priority flexible
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 2. 创建并激活环境
conda create -n openvla python=3.10 -y
conda activate openvla
```

## 3. 安装核心计算库

避开 NGC 系统源干扰，强制安装兼容 Flash Attention 的 CUDA 12.1 版本 PyTorch。

```bash
pip install uv

uv pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

现在：Conda 环境 (openvla) 已激活，PyTorch 已就位。请依次执行以下步骤，完成 OpenVLA 和 LIBERO 的最终部署。

### 第一步：安装仿真环境与 OpenVLA 源码

```bash
cd /datadisk

# 1. 创建项目文件夹
mkdir -p my_project/openvla_root
cd my_project

# 2. 安装 LIBERO 和 MuJoCo
uv pip install mujoco
# pip install libero 下来的版本可能很老
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git --depth=1
# 使用加速服务：
git clone https://ghfast.top/https://github.com/Lifelong-Robot-Learning/LIBERO.git --depth=1
cd LIBERO
uv pip install -r requirements.txt
uv pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
# Then install the libero package:
pip install -e .
cd ..

# 3. 克隆 OpenVLA 代码
git clone https://github.com/openvla/openvla.git --depth=1
# 使用加速服务
git clone https://ghfast.top/https://github.com/openvla/openvla.git --depth=1
cd openvla

# 4. 安装其余依赖
uv pip install -r requirements-min.txt
uv pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

**安装完 libero 会出现错误：**

```
ModuleNotFoundError: No module named 'libero'
```

**问题表现为：**

在 LIBERO 源码目录下可以正常 `import libero`，但只要切换到其他目录（例如 `cd ..`），就会立刻报错找不到该模块。

此外，即使运行 `pip install -e .` 终端提示安装成功，但进入 Python 执行 `import libero; print(libero.__file__)` 会返回 `None`，且检查其路径 `libero.__path__` 会显示它是一个 `_NamespacePath`。

**产生原因：**

LIBERO 源码的 `libero` 目录下缺少了 `__init__.py` 文件。在 Python 3 中，缺少该文件的目录会被当作“命名空间包（Namespace Package）”处理，而不是一个标准的 Python 包。这导致 `setup.py`（或相关的构建工具）无法正确识别和注册包的内容，所谓的“安装成功”仅仅是记录了名号，并没有真正将代码链接到 Python 环境的 `site-packages` 搜索路径中。

**如何解决：**

手动补齐缺失的 `__init__.py` 文件并重新安装即可：

```bash
# 1. 确保进入 LIBERO 源码根目录
cd /datadisk/my_project/LIBERO

# 2. 创建缺失的 __init__.py 文件
touch libero/__init__.py

# 3. 重新以可编辑模式安装
pip install -e .
```

**验证是否彻底解决：**

必须**离开**源码目录（例如 `cd ..`），然后进入 Python 环境测试：

```bash
python -c "import libero; print('Success! File:', libero.__file__)"
```

如果成功输出了带有 `__init__.py` 的完整路径（不再是 `None`），说明模块已经正确挂载到了全局环境中。

>   在 GitHub 的一个 [issue](https://github.com/huggingface/lerobot/issues/2452) 中看到似乎运行 `pip install -e ".[libero]"` 可以解决，但我并未尝试

### 第二步：安装 Flash Attention 2 (A800 核心加速)

这是最耗时的一步（通常需要 5-10 分钟编译），请耐心等待不要中断。

```bash
uv pip install psutil ninja packaging
uv pip install flash-attn --no-build-isolation
```

>   实际上消耗了大概 30 min

### 第三步： 登录 huggingface 下载模型权重

我们需要去官网生成对应token，注意**不能使用香港节点进行注册**，注册好后认证邮箱，接下来可以去settings中生成token并复制下来，这样就获得了后续下载的权限。

```bash
pip install huggingface_hub
# 连接国内镜像站
export HF_ENDPOINT=https://hf-mirror.com
# 输入刚刚生成的 token
# 遇到 Add token as git credential? (Y/n) 时，输入 n 并回车
hf auth login 
```

### 第四步： 验证

登录成功后，创建一个[脚本](test_env.py)，一次性验证 Flash Attention (GPU加速)、LIBERO (仿真) 和 OpenVLA (模型加载) 是否全部正常。

```bash
touch /datadisk/my_project/test_env.py
```

在编辑器中粘贴以下代码：

```python
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
```

补齐 python 包，设置渲染环境变量后运行脚本：

```bash
# 1. 补齐缺失的 Python 包
pip install accelerate

# 2. 设置渲染环境变量
export PYOPENGL_PLATFORM=egl

# 3. 再次运行测试脚本
python test_env.py
```

此时大概率虽然模型好了，但日志里还有一行红字： 

```bash
LIBERO Error: ‘NoneType’ object has no attribute ‘eglQueryString’ ❌
```

这意味着仿真环境（机器人模拟器）还是坏的。虽然模型加载了，但机器人“看不见”东西。这是因为我们是租的卡，服务器没有显示器，我们需要安装一个支持“无头模式”渲染的库。

故接下来请执行如下指令完善环境：

```bash
# 1. 安装缺少的系统级 EGL 库 (解决报错的核心)
apt-get update && apt-get install -y libegl1 libegl1-mesa

# 2. 将渲染配置写入系统文件 (防止下次重启失效)
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
conda activate openvla

# 3. 再次验证
python test_final.py
```

此时会出现提示 `Do you want to specify a custom path for the dataset folder? (Y/N):`

我们选择 `Y`，接着它会问你路径，请输入：

```bash
/datadisk/my_project/libero_data
```

后续的提示也一律输入Y即可。

## 4. 运行 demo

由于服务器是无头环境（无显示器），无法实时查看机器人状态。本节目标是运行推理脚本，让 OpenVLA 控制机器人执行任务，并导出视频用于离线观察。

### 第一步：创建推理脚本

我们先创建 `quick_start.py`，用于完成以下流程：

1. 启动 LIBERO 仿真环境（任务：拿起黑色碗并放到盘子上）
2. 加载 OpenVLA 模型
3. 循环执行“观测 → 预测动作 → 执行动作”
4. 保存 `.mp4` 视频

请在终端执行：

```bash
touch /datadisk/my_project/quick_start.py
```

将下列内容脚本 [quick_start.py](quick_start.py)：

```python
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
```

### 第二步：安装视频处理库

安装 `imageio` 用于将帧合成为视频：

```bash
pip install imageio[ffmpeg]
```

### 第三步： 运行脚本

每次运行前都要先激活环境并加载变量：

```bash
source ~/.bashrc
conda activate openvla
python quick_start.py
```

若报错 `Error: BDDL_FILES environment variable is missing...`，先检查 `BDDL_FILES`：

```bash
echo $BDDL_FILES
```

若无输出，说明该变量未生效。请找到服务器上 `LIBERO/libero/libero/bddl_files` 的绝对路径，并写入 `~/.bashrc`：

```bash
# 请将下面的路径替换为您实际存放 bddl_files 的绝对路径
echo 'export BDDL_FILES="/datadisk/my_project/LIBERO/libero/libero/bddl_files"' >> ~/.bashrc

# 重新应用配置并再次确认
source ~/.bashrc
conda activate openvla
python quick_start.py
```

### 第四步：查看视频

由于服务器无显示器，需要将生成的 `robot_demo.mp4` 下载到本地查看。

若你使用 VS Code 远程连接，可按以下步骤下载：

1. 在左侧资源管理器中进入 `/datadisk/my_project`
2. 找到 `robot_demo.mp4`
3. 右键选择 `Download`
4. 保存到本地后直接播放

### 第五步：问题复现与改进版脚本

实际运行中常见两个问题：

1. **视频太短（仅 30 步）**：机械臂可能还没完成明显动作，视频就结束。
2. **重连后加载失败**：终端断开或网络不稳定时，模型可能再次联网拉取资源，导致推理启动失败。

因此建议创建 `quick_start_offline.py`，做两处改动：

- 在模型与处理器加载时加入 `local_files_only=True`（强制离线）
- 增加推理步数与视频帧率（100 步，`fps=20`）

如果不做这两处修改，典型结果是：**视频可观察性差**，以及**重启后脚本偶发卡在模型加载/联网阶段**。

新建脚本 [quick_start_offline.py](quick_start_offline.py)：

```bash
touch quick_start_offline.py
```

粘贴下列内容：

```python
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
UNNORM_KEY = "bridge_orig" 

print("="*50)
print("[1/4] Initializing LIBERO Simulation Environment...")

if "BDDL_FILES" not in os.environ:
    raise ValueError("❌ Error: BDDL_FILES environment variable is missing.")

env_args = {
    "bddl_file_name": os.path.join(os.environ["BDDL_FILES"], BDDL_FOLDER, BDDL_FILE),
    "camera_heights": 256,
    "camera_widths": 256,
    "camera_depths": False,
}

try:
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    print(f"✅ Simulation Ready.")
except Exception as e:
    print(f"❌ Failed to load environment: {e}")
    exit()

print("\n[2/4] Loading OpenVLA Model (OFFLINE MODE)...")

# 关键改动 1：强制离线加载，避免断网后失败
try:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        local_files_only=True  # <--- 强制离线
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True  # <--- 强制离线
    ).to(DEVICE)
    print("✅ Model Loaded from local cache.")
except Exception as e:
    print(f"❌ Failed to load model offline. Ensure model is downloaded. Error: {e}")
    exit()

print("\n[3/4] Starting Robot Inference Loop (100 steps demo)...")

prompt = "In: What action should the robot take to {}?\nOut:"
instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"

frames = [] 
obs = env.reset()

# 关键改动 2：增加步数，提升视频可观察性
for step in range(100): 
    img_array = obs["agentview_image"] 
    image = Image.fromarray(img_array) 
    frames.append(img_array[::-1]) 

    inputs = processor(text=prompt.format(instruction), images=image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

    if step % 10 == 0:
        print(f"Step {step}: Action XYZ = {action[:3]}")

    # 调试期动作放大（仅用于观察动作变化）
    action = action * 10.0 

    obs, reward, done, info = env.step(action)
    print(f"\rStep {step+1}/100", end="", flush=True)

print("\n\n[4/4] Saving Video...")
video_path = "robot_demo_debug.mp4"
imageio.mimsave(video_path, np.stack(frames), fps=20)
print(f"🎉 Video saved to: {os.path.abspath(video_path)}")
print("="*50)
```

重新运行后查看 `robot_demo_debug.mp4`：

```bash
source ~/.bashrc
conda activate openvla
python quick_start_offline.py
```

若视频中出现“动作很小、几乎不动”或“动作不稳定”，通常说明基础模型在该任务上的泛化不足，下一步就需要进入微调流程。

## 容易出现的问题

- 使用 `uv pip` 进行安装，解决速度慢的问题（尝试使用代理、双机器传输未果）；
- 安装 LIBERO 时 pip 搜索不到的问题，最终通过创建 `__init__.py` 脚本解决；
- 解决服务器无头显示的问题；
- 指定 torch 版本以确保兼容性。
