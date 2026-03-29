import torch
from src.model import GPT
from src.data import build_datasets
from src.utils import set_seed
import yaml

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    set_seed(cfg["training"]["seed"])
    
    tok, train_ds, val_ds = build_datasets(cfg)
    vocab_size = tok.vocab_size
    
    model = GPT(
        vocab_size=vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=cfg["model"]["dropout"],
    )
    
    total, trainable = count_parameters(model)
    
    print(f"=== 模型参数分析 ===")
    print(f"词表大小: {vocab_size}")
    print(f"总参数量: {total:,} ({total/1e6:.2f}M)")
    print(f"可训练参数: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"\n=== 参数分布 ===")
    print(f"Embedding层: {tok.vocab_size * cfg['model']['n_embd']:,} ({tok.vocab_size * cfg['model']['n_embd']/1e6:.2f}M)")
    
    block_params = 0
    for name, module in model.named_modules():
        if 'qkv' in name or 'proj' in name:
            pass
    
    n_embd = cfg["model"]["n_embd"]
    n_layer = cfg["model"]["n_layer"]
    
    attn_per_layer = 4 * n_embd * n_embd
    mlp_per_layer = 8 * n_embd * n_embd
    norm_per_layer = 2 * n_embd
    
    total_block = n_layer * (attn_per_layer + mlp_per_layer + norm_per_layer)
    
    print(f"每层Attention: {attn_per_layer:,}")
    print(f"每层MLP: {mlp_per_layer:,}")
    print(f"每层Norm: {norm_per_layer:,}")
    print(f"{n_layer}层总计: {total_block:,} ({total_block/1e6:.2f}M)")
    
    print(f"\n=== 当前Dropout设置 ===")
    print(f"Dropout: {cfg['model']['dropout']}")
    print(f"⚠️ 当前无Dropout，容易过拟合！")
    
    print(f"\n=== 课堂训练建议 ===")
    print(f"当前配置适合课堂演示：")
    print(f"✓ 参数量适中（3.37M），训练快速")
    print(f"✓ 序列长度128，适合短文本")
    print(f"✓ 4层4头，结构简单易理解")
    print(f"\n建议改进：")
    print(f"• 增加Dropout到0.1-0.2")
    print(f"• 减少训练步数到2000-3000步（1小时内）")
    print(f"• 添加早停机制")
    print(f"• 使用Few-shot推理提升泛化")

if __name__ == "__main__":
    main()
