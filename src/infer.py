import os
import json
import argparse
import warnings
import torch

from .model import GPT
from .data import build_datasets
from .tokenizer import load_tokenizer

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def load_checkpoint(path):
    if not os.path.exists(path):
        print(f"错误：未找到模型文件 {path}")
        print("请先运行训练：uv run python -m src.train")
        raise SystemExit(1)
    if path.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise ImportError("请安装 safetensors: pip install safetensors")
        state_dict = load_safetensors(path)
        config_path = path.replace(".safetensors", ".json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = None
        return {"model": state_dict, "cfg": cfg}
    return torch.load(path, map_location="cpu", weights_only=False)

PUNCT = set(list(",，。．、：:；;！!？?…"))

def _is_punct_token(tok, tid: int) -> bool:
    try:
        s = tok.decode([tid])
        return len(s) > 0 and s[0] in PUNCT
    except Exception:
        return False

def _trim_leading_punct(s: str) -> str:
    i = 0
    while i < len(s) and (s[i].isspace() or s[i] in PUNCT):
        i += 1
    return s[i:]

def generate(model, tok, prompt, max_new_tokens=64, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty: float = 1.0, stop_strings=None, min_tokens: int = 5, device=None, return_confidence=False):
    model.eval()
    # normalize prompt: collapse or remove spaces commonly inserted in Chinese
    norm = prompt.replace(" ", "").replace("\u3000", "")
    # 手动添加 BOS，不添加 EOS
    prefix = [tok.bos_id] + tok.encode("用户:" + norm + "\n助手:", add_special_tokens=False)
    x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)

    recent = []
    kv_caches = None
    token_probs = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # 推理优化：如果是第一步，输入全量 prefix；后续步仅输入上一个生成的 token + KV Cache
            if step == 0:
                cur_input = x
            else:
                cur_input = x[:, -1:]

            logits, kv_caches = model(cur_input, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(1e-6, temperature if temperature > 0 else 1.0)

            # 基础屏蔽逻辑：无论温度如何都生效
            if hasattr(tok, 'pad_id') and tok.pad_id is not None and tok.pad_id >= 0:
                logits[0, tok.pad_id] = -float('inf')
            if hasattr(tok, 'bos_id') and tok.bos_id is not None and tok.bos_id >= 0:
                logits[0, tok.bos_id] = -float('inf')
            if hasattr(tok, 'unk_id') and tok.unk_id is not None and tok.unk_id >= 0:
                logits[0, tok.unk_id] = -float('inf')

            # 强制最小生成长度，防止秒断
            if step < min_tokens and hasattr(tok, 'eos_id') and tok.eos_id is not None and tok.eos_id >= 0:
                logits[0, tok.eos_id] = -float('inf')

            # 重复惩罚：无论 greedy 还是采样都生效
            if repetition_penalty > 1.0 and len(recent) > 0:
                for tid in recent[-32:]:
                    logits[0, tid] = logits[0, tid] / repetition_penalty

            # 计算 softmax 概率（用于置信度）
            all_probs = torch.softmax(logits, dim=-1)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = all_probs.clone()
                if top_k > 0:
                    v, i = torch.topk(probs, top_k)
                    p = torch.zeros_like(probs).scatter_(1, i, v)
                    s = p.sum(dim=-1, keepdim=True)
                    probs = torch.where(s > 0, p / s, probs)
                if top_p < 1.0:
                    srt, idx = torch.sort(probs, descending=True)
                    c = torch.cumsum(srt, dim=-1)
                    m = c <= top_p
                    srt = srt * m
                    p = torch.zeros_like(probs).scatter_(1, idx, srt)
                    s = p.sum(dim=-1, keepdim=True)
                    probs = torch.where(s > 0, p / s, probs)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                if probs.sum() == 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    next_id = torch.multinomial(probs, 1)

            # 收集置信度
            if return_confidence:
                token_probs.append(all_probs[0, next_id.item()].item())

            x = torch.cat([x, next_id], dim=1)
            recent.append(next_id.item())
            if next_id.item() == tok.eos_id:
                break
            if stop_strings:
                out_ids = x[0].tolist()[len(prefix):]
                out_text = tok.decode(out_ids)
                if any(out_text.endswith(ss) for ss in stop_strings):
                    break
    out_ids = x[0].tolist()[len(prefix):]
    text = _trim_leading_punct(tok.decode(out_ids))
    if return_confidence:
        avg_conf = sum(token_probs) / len(token_probs) if token_probs else 0.0
        return {"text": text, "avg_confidence": avg_conf, "token_probs": token_probs}
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yml", help="配置文件路径")

    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.5)
    ap.add_argument("--stop_strings", nargs='*', default=[])
    ap.add_argument("--show_label", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"]) 
    args = ap.parse_args()
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
        n_kv_head=cfg["model"].get("n_kv_head"),
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
    
    # 预热推理逻辑
    print("--- 模型推理结果 ---")
    text = generate(m, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, args.stop_strings, device=device)
    if args.show_label:
        print("回答:" + text)
    else:
        print(text)

if __name__ == "__main__":
    main()
