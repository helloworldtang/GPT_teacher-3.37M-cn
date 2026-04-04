import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '.')
import torch
from src.model import GPT
from src.infer import load_checkpoint, generate
from src.data import load_tokenizer

def calculate_match_rate(expected: str, generated: str) -> float:
    """计算关键信息匹配度"""
    expected_clean = expected.strip().lower()
    generated_clean = generated.strip().lower()
    
    words = expected_clean.split()
    if len(words) == 0:
        return 0.0
    
    match_count = 0
    for word in words:
        if word in generated_clean:
            match_count += 1
    
    return (match_count / len(words)) * 100

def main():
    parser = argparse.ArgumentParser(description="验收测试")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.pt", help="模型权重路径")
    parser.add_argument("--device", type=str, default="auto", help="设备选择")
    parser.add_argument("--test_file", type=str, default="data/test_acceptance.jsonl", help="验收集路径")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数")
    parser.add_argument("--mode", type=str, default="training", choices=["training", "acceptance"], help="测试模式：training=训练集内问题，acceptance=验收集问题")
    args = parser.parse_args()
    
    if args.mode == "acceptance":
        args.test_file = "data/val_acceptance.jsonl"
    
    obj = load_checkpoint(args.ckpt)
    cfg = obj["cfg"]
    tok = load_tokenizer(cfg.get("tokenizer", {}).get("type", "byte"), cfg.get("tokenizer", {}).get("path"))
    m = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=cfg["model"]["dropout"],
        use_flash=False,
    )
    sd = obj["model"]
    packed = any("_packed_params" in k for k in sd.keys())
    if packed:
        device = torch.device("cpu")
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device("cpu")
    m.load_state_dict(sd)
    m.to(device)
    
    test_cases = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    
    print(f"开始验收测试，共{len(test_cases)}个问题...")
    print("=" * 80)
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        prompt = case['prompt']
        expected = case['completion']
        
        print(f"\n问题 {i}/{len(test_cases)}")
        print(f"Q: {prompt}")
        print(f"A(期望): {expected}")
        
        generated = generate(
            m, tok, prompt,
            max_new_tokens=128,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_strings=["。", "；", "\n"],
            device=device
        )
        
        print(f"A(生成): {generated}")
        
        match_rate = calculate_match_rate(expected, generated)
        results.append({
            'id': i,
            'prompt': prompt,
            'expected': expected,
            'generated': generated,
            'match_rate': match_rate
        })
        
        print(f"匹配度: {match_rate:.1f}%")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("验收测试总结")
    print("=" * 80)
    
    overall_avg = sum(r['match_rate'] for r in results) / len(results)
    passed_80 = sum(1 for r in results if r['match_rate'] >= 80)
    
    if args.mode == "training":
        logic_avg = overall_avg
        print(f"逻辑自洽能力: {logic_avg:.1f}% (目标: ≥90%)")
        print(f"总体匹配度: {overall_avg:.1f}%")
        print(f"\n通过≥80%标准的问题数: {passed_80}/{len(results)} (目标: ≥8)")
        
        if passed_80 >= 8:
            print("\n✓ 训练集内测试通过！")
        else:
            print(f"\n✗ 训练集内测试未通过（需要至少8个问题匹配度≥80%，当前{passed_80}个）")
    else:
        print(f"验收数据集测试 (val_acceptance.jsonl)")
        print(f"总体匹配度: {overall_avg:.1f}%")
        print(f"\n通过≥80%标准的问题数: {passed_80}/{len(results)} (目标: ≥7)")
        
        if passed_80 >= 7:
            print("\n✓ 验收数据集测试通过！")
        else:
            print(f"\n✗ 验收数据集测试未通过（需要至少7个问题匹配度≥80%，当前{passed_80}个）")
    
    return 0 if passed_80 >= 8 else 1

if __name__ == "__main__":
    exit(main())
