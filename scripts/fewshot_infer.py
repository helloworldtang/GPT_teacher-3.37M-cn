import argparse
import torch
import yaml
import json
from src.model import GPT
from src.utils import set_seed
from src.data import build_datasets
import random

def load_model(checkpoint_path, device, cfg):
    model = GPT(
        vocab_size=cfg["model"]["vocab_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=cfg["model"]["dropout"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def sample_fewshot_examples(train_data, n_examples=3, exclude_question=None):
    if exclude_question:
        available = [item for item in train_data if item["prompt"] != exclude_question]
    else:
        available = train_data
    
    if len(available) < n_examples:
        return available
    
    return random.sample(available, n_examples)

def generate_fewshot_prompt(question, examples):
    prompt_parts = []
    for example in examples:
        prompt_parts.append(f"用户:{example['prompt']}")
        prompt_parts.append(f"助手:{example['completion']}")
    
    prompt_parts.append(f"用户:{question}")
    prompt_parts.append("助手:")
    
    return "\n".join(prompt_parts)

def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8, top_p=0.9):
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    x = torch.tensor(tokens, device=device).unsqueeze(0)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            next_token_logits = logits[:, -1, :] / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_id:
                break
            
            generated.append(next_token.item())
            x = torch.cat([x, next_token], dim=1)
    
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--n_examples", type=int, default=3, help="Few-shot examples数量")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    cfg = yaml.safe_load(open("config.yaml", "r"))
    tok, train_ds, val_ds = build_datasets(cfg)
    
    cfg["model"]["vocab_size"] = tok.vocab_size
    
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, cfg)
    
    train_data = []
    with open("data/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    
    examples = sample_fewshot_examples(train_data, args.n_examples, args.prompt)
    prompt = generate_fewshot_prompt(args.prompt, examples)
    
    print(f"=== Few-Shot推理 (n={args.n_examples}) ===")
    print(f"Prompt长度: {len(prompt)} 字符")
    print(f"问题: {args.prompt}")
    print(f"\n=== 使用的Few-Shot示例 ===")
    for i, example in enumerate(examples):
        print(f"\n示例 {i+1}:")
        print(f"  问题: {example['prompt']}")
        print(f"  答案: {example['completion'][:50]}{'...' if len(example['completion']) > 50 else ''}")
    
    print(f"\n=== 生成结果 ===")
    answer = generate(model, tok, prompt, args.max_tokens, args.temperature, args.top_p)
    print(answer)
    
    print(f"\n=== 对比Zero-Shot ===")
    zero_shot_prompt = f"用户:{args.prompt}\n助手:"
    zero_shot_answer = generate(model, tok, zero_shot_prompt, args.max_tokens, args.temperature, args.top_p)
    print(f"Zero-Shot: {zero_shot_answer}")

if __name__ == "__main__":
    main()
