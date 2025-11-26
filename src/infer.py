import os
import argparse
import warnings
import torch

from .model import GPT
from .data import build_datasets
from .tokenizer import load_tokenizer

# suppress specific noisy warnings from torch during quantized ckpt load
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


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


def generate(
    model,
    tok,
    prompt,
    max_new_tokens=64,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty: float = 1.0,
    stop_strings=None,
    min_tokens: int = 5,
    device=None,
):
    model.eval()
    # normalize prompt: collapse or remove spaces commonly inserted in Chinese
    norm = prompt.replace(" ", "").replace("\u3000", "")
    prefix = tok.encode("用户:" + norm + "\n助手:", add_special_tokens=True)
    x = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    recent = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if hasattr(tok, "pad_id") and tok.pad_id is not None and tok.pad_id >= 0:
                logits[0, tok.pad_id] = -float("inf")
            if hasattr(tok, "bos_id") and tok.bos_id is not None and tok.bos_id >= 0:
                logits[0, tok.bos_id] = -float("inf")
            if hasattr(tok, "unk_id") and tok.unk_id is not None and tok.unk_id >= 0:
                logits[0, tok.unk_id] = -float("inf")
            if (
                step == 0
                and hasattr(tok, "eos_id")
                and tok.eos_id is not None
                and tok.eos_id >= 0
            ):
                logits[0, tok.eos_id] = -float("inf")
            if repetition_penalty > 1.0 and len(recent) > 0:
                for tid in recent[-16:]:
                    logits[0, tid] = logits[0, tid] / repetition_penalty
            if (
                step < min_tokens
                and hasattr(tok, "eos_id")
                and tok.eos_id is not None
                and tok.eos_id >= 0
            ):
                logits[0, tok.eos_id] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
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
            x = torch.cat([x, next_id], dim=1)
            recent.append(next_id.item())
            if next_id.item() == tok.eos_id:
                break
            if stop_strings:
                out_ids = x[0].tolist()[len(prefix) :]
                out_text = tok.decode(out_ids)
                if any(out_text.endswith(ss) for ss in stop_strings):
                    break
    out_ids = x[0].tolist()[len(prefix) :]
    return _trim_leading_punct(tok.decode(out_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--stop_strings", nargs="*", default=None)
    ap.add_argument("--show_label", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    dev_arg = args.device.lower()
    if dev_arg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif dev_arg == "cpu":
        device = torch.device("cpu")
    else:
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    obj = load_checkpoint(args.ckpt, device if device.type != "cpu" else "cpu")
    cfg = obj["cfg"]
    tok = load_tokenizer(
        cfg.get("tokenizer", {}).get("type", "byte"),
        cfg.get("tokenizer", {}).get("path"),
    )
    m = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=cfg["model"]["dropout"],
    )
    sd = obj["model"]
    packed = any("_packed_params" in k for k in sd.keys())
    dev_arg = args.device.lower()
    if packed:
        device = torch.device("cpu")
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        m.load_state_dict(sd)
    else:
        m.load_state_dict(sd)
        m = m.float().to(device)
        m.tok_emb = m.tok_emb.to(device)
        m.tok_emb.weight.data.copy_(sd["tok_emb.weight"].to(device))
        m.head = m.head.to(device)
        m.head.weight = torch.nn.Parameter(
            sd["tok_emb.weight"].detach().clone().to(device)
        )
    text = generate(
        m,
        tok,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        args.stop_strings,
        device=device,
    )
    if args.show_label:
        print("回答:" + text)
    else:
        print(text)


if __name__ == "__main__":
    main()
