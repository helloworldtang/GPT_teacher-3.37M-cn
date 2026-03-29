"""
将训练好的 .pt 权重转换为 safetensors 格式

safetensors 是生产环境部署的推荐格式：
- 安全：避免 pickle 反序列化漏洞
- 快速：mmap 零拷贝加载
- 兼容：Hugging Face 生态标准格式

用法:
    python -m src.export_safetensors --ckpt checkpoints/last.pt --output checkpoints/model.safetensors
    python -m src.export_safetensors --ckpt checkpoints/last.pt --output checkpoints/model.safetensors --include_config
"""

import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file


def convert_pt_to_safetensors(ckpt_path: str, output_path: str, include_config: bool = False):
    obj = torch.load(ckpt_path, map_location="cpu")
    
    state_dict = obj["model"]
    
    tensors = {}
    seen_storage = {}
    for key, value in state_dict.items():
        if "_packed_params" in key:
            print(f"跳过量化参数: {key}")
            continue
        if isinstance(value, torch.Tensor):
            storage_ptr = value.storage().data_ptr()
            if storage_ptr in seen_storage:
                existing_key = seen_storage[storage_ptr]
                print(f"权重共享: {key} -> {existing_key} (保存一份)")
                tensors[key] = tensors[existing_key].clone()
            else:
                seen_storage[storage_ptr] = key
                tensors[key] = value.contiguous()
    
    save_file(tensors, output_path)
    print(f"已保存: {output_path}")
    print(f"参数数量: {len(tensors)}")
    
    if include_config:
        cfg = obj.get("cfg", {})
        config_path = Path(output_path).with_suffix(".json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"已保存配置: {config_path}")
    
    return len(tensors)


def main():
    parser = argparse.ArgumentParser(description="转换 .pt 为 safetensors 格式")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.pt", help="输入的 .pt 权重路径")
    parser.add_argument("--output", type=str, default="checkpoints/model.safetensors", help="输出的 safetensors 路径")
    parser.add_argument("--include_config", action="store_true", help="同时导出配置为 JSON")
    args = parser.parse_args()
    
    convert_pt_to_safetensors(args.ckpt, args.output, args.include_config)


if __name__ == "__main__":
    main()
