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