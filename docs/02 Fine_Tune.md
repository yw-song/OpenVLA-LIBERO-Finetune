# OpenVLA  微调

## 一、训练

在 `/datadisk/my_project/` 下创建一个名为 [`download_data.py`](download_data.py) 的文件，代码如下：

```python
import os
from huggingface_hub import snapshot_download

# 数据保存路径 (建议放在 datadisk 这种大盘里)
DATA_ROOT = "/datadisk/datasets/openvla-libero-spatial"
os.makedirs(DATA_ROOT, exist_ok=True)

print(f"正在准备下载数据到: {DATA_ROOT} ...")

# 下载 LIBERO-Spatial 数据集 (针对空间方位理解的任务)
# 这是一个很好的起点，数据量适中 (约 15GB)
snapshot_download(
    repo_id="openvla/modified_libero_rlds", 
    repo_type="dataset",
    local_dir=DATA_ROOT,
    allow_patterns="*libero_spatial*",  # 只下载 spatial 相关数据，节省时间
    resume_download=True
)

print("✅ 下载完成！Dataset Download Complete!")
```

下载并安装 `dlimp`（处理 RLDS 数据集的必备库）：

>   使用学术加速

```bash
git clone https://ghfast.top/https://github.com/kvablack/dlimp.git --depth=1
cd dlimp
pip install -e .
cd ..
HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com) python download_data.py 
```

## 二、训练脚本配置

在 `/datadisk/my_project/openvla/` 目录下，新建一个文件，命名为 [`train_libero.sh`](train_libero.sh)，内容如下（注意在脚本倒数第三行修改为你自己的 `wandb_entity` 用户名）：

```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 显存优化配置：如果你不想登录 wandb，可以取消下面这行的注释
# export WANDB_MODE=offline

# 设置多卡/单卡端口，防止冲突
export MASTER_PORT=29505

# 启动训练
# A800 专属配置：
# 1. 不使用 --load_in_4bit (保持 BFloat16 高精度，效果更好)
# 2. batch_size 设为 16 (显存大，跑得更快)

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/datadisk/datasets/openvla-libero-spatial" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "/datadisk/checkpoints" \
  --adapter_tmp_dir "/datadisk/adapter_tmp" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla-finetune" \
  --wandb_entity "YOUR_WANDB_USERNAME" \  # <--- 请在这里填入你的 WandB 用户名
  --save_steps 1000 \
  --max_steps 5000

```

## 三、环境配置与防爆雷安装 (极其重要)

>   **⚠️ 警告**：大模型环境对版本极其敏感。请严格执行以下版本锁定命令，**切勿盲目使用 `--upgrade`**，否则会导致 PyTorch 版本错乱、NumPy 冲突以及 FlashAttention 编译失败。

首先，进入项目根目录并赋予脚本执行权限：

```bash
cd /datadisk/my_project/openvla/
chmod +x train_libero.sh
```

**第 1 步：锁定 PyTorch 及底层计算生态 (防 Numpy 冲突)**

```bash
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
uv pip install "numpy<2"
```

**第 2 步：锁定 OpenVLA 兼容的模型库与显存优化库**

```bash
uv pip install transformers==4.40.1 tokenizers==0.19.1
uv pip install bitsandbytes==0.43.1
uv pip install draccus peft wandb accelerate
```

**第 3 步：注册本项目并重新编译 FlashAttention**

```bash
# 将当前 prismatic 包注册到环境
pip install -e .

# 暴力卸载并重新编译 FlashAttention (必须带 --no-build-isolation 以匹配当前 PyTorch 版本)
pip uninstall -y flash-attn
pip install flash-attn --no-build-isolation
```

## 四、启动训练与监控

1.  **登录 WandB**（用于监控 Loss 曲线）：

    前往 [wandb.ai官网](https://wandb.ai/site) 注册并获取你个人的 API Key。

    在终端输入：

    ```bash
    wandb login
    ```

    根据提示粘贴你的 API Key（出于安全考虑，密码输入时在屏幕上不可见，粘贴后直接回车即可

2.  **运行训练！**

    ```bash
    ./train_libero.sh
    ```

    >   注：启动初期会出现几行 TensorFlow 找不到 TensorRT 或 cuDNN 重复注册的报错，属于正常现象，耐心等待进度条加载即可。

## 五、微调效果验证

在 `/datadisk/my_project` 下创建验证脚本 [`test_model.py`](test_model.py)，用于快速检查微调后的模型是否可正常推理：

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# 1. 微调后完整模型路径
MODEL_PATH = "/datadisk/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug" 

print(f"正在从本地加载模型与处理器: {MODEL_PATH}")

# 2. 从本地路径加载处理器（Processor）
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 3. 从本地路径加载完整模型（使用 bfloat16 节省显存）
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda")

print("模型加载成功！准备测试...")

# 4. 构造测试图片
print("⚠️ 由于网络限制，正在使用本地生成的纯黑测试图片...")
# 创建一张 224x224 的纯黑图片，用于验证推理链路是否打通
image = Image.new('RGB', (224, 224), color='black')

# 5. 输入一条指令（Prompt）
prompt = "In: What action should the robot take to [pick up the yellow block]?\nOut:"

# 6. 处理输入
# 注意：确保 processor 输出的 tensor 与模型类型一致（bfloat16）
inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

# 7. 预测动作
print("正在预测动作...")
# ⚠️ 提示：unnorm_key 用于动作反归一化，通常需要与训练数据集保持一致。
# 若报错找不到 "bridge_orig" 的统计数据，可改为你实际训练的数据集名（如 "libero_spatial"）
action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print("\n====== 预测结果 ======")
print(f"指令: {prompt}")
print(f"模型生成的动作向量: {action}")
print("======================")
```

保存后在终端执行：

```bash
HF_ENDPOINT=https://hf-mirror.com python test_model.py  # 使用 Hugging Face 国内镜像
```

当终端打印出动作向量后，说明模型已可用：前 3 个数表示机械臂空间位移（x, y, z），中间 3 个数表示旋转角（roll, pitch, yaw），最后 1 个数表示夹爪开合状态（`0.` 通常表示闭合或保持原状）。

由于本次使用的是 `libero_spatial` 数据集微调，更有参考价值的验证方式是运行 Libero 模拟器评估并查看任务成功率。先定位评估脚本：

```bash
find /datadisk/my_project -name run_libero_eval.py
```

通常会得到类似结果（后续请替换为你机器上的真实路径）：

```bash
/datadisk/my_project/openvla/experiments/robot/libero/run_libero_eval.py
```

然后带上正确脚本路径与训练后的 checkpoint 运行评估（同时设置 `HF_ENDPOINT`，可避免模型加载时的联网问题）：

```bash
HF_ENDPOINT=https://hf-mirror.com python /datadisk/my_project/openvla/experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /datadisk/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name libero_spatial \
  --center_crop True
```

Libero 模拟器评估大约需要 3 小时左右，最终结果可在运行日志与 WandB（若已开启）中查看。
