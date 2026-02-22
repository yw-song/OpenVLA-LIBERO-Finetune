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