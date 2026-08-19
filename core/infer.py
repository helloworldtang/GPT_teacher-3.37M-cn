"""推理引擎：checkpoint 加载与带 KV Cache 的自回归生成。"""

import argparse
import json
import os
import warnings
from typing import Any, Literal, TypedDict, overload

import torch

from core.model import GPT, KvCache
from core.tokenizer import TokenizerLike, load_tokenizer

try:
    from safetensors.torch import load_file as load_safetensors

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

PUNCT = set(",，。．、：:；;！!？?…")


class GenerateResult(TypedDict):
    """带置信度的生成结果。"""

    text: str
    avg_confidence: float
    token_probs: list[float]


def load_checkpoint(path: str) -> dict[str, Any]:
    """加载 checkpoint（.pt 或 .safetensors）。

    Args:
        path: checkpoint 文件路径。

    Returns:
        至少包含 "model"（state_dict）与 "cfg"（配置）的字典。

    Raises:
        SystemExit: 文件不存在。
        ImportError: safetensors 格式但未安装 safetensors。
    """
    if not os.path.exists(path):
        print(f"错误：未找到模型文件 {path}")
        print("请先运行训练：uv run python -m train.train")
        raise SystemExit(1)
    if path.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise ImportError("请安装 safetensors: pip install safetensors")
        state_dict = load_safetensors(path)
        config_path = path.replace(".safetensors", ".json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                cfg: dict[str, Any] | None = json.load(f)
        else:
            cfg = None
        return {"model": state_dict, "cfg": cfg}
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def _is_punct_token(tok: TokenizerLike, tid: int) -> bool:
    """判断 token 解码后是否以标点开头。"""
    try:
        s = tok.decode([tid])
        return len(s) > 0 and s[0] in PUNCT
    except Exception:
        return False


def _trim_leading_punct(s: str) -> str:
    """去掉开头的空白与标点。"""
    i = 0
    while i < len(s) and (s[i].isspace() or s[i] in PUNCT):
        i += 1
    return s[i:]


@overload
def generate(
    model: GPT,
    tok: TokenizerLike,
    prompt: str,
    *,
    max_new_tokens: int = ...,
    temperature: float = ...,
    top_k: int = ...,
    top_p: float = ...,
    repetition_penalty: float = ...,
    stop_strings: list[str] | None = ...,
    min_tokens: int = ...,
    device: torch.device | str | None = ...,
    wrap_prompt: bool = ...,
    return_confidence: Literal[False] = ...,
) -> str: ...


@overload
def generate(
    model: GPT,
    tok: TokenizerLike,
    prompt: str,
    *,
    max_new_tokens: int = ...,
    temperature: float = ...,
    top_k: int = ...,
    top_p: float = ...,
    repetition_penalty: float = ...,
    stop_strings: list[str] | None = ...,
    min_tokens: int = ...,
    device: torch.device | str | None = ...,
    wrap_prompt: bool = ...,
    return_confidence: Literal[True],
) -> GenerateResult: ...


def generate(
    model: GPT,
    tok: TokenizerLike,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_strings: list[str] | None = None,
    min_tokens: int = 5,
    device: torch.device | str | None = None,
    wrap_prompt: bool = True,
    return_confidence: bool = False,
) -> str | GenerateResult:
    """自回归生成回答，首步后仅输入增量 token + KV Cache。

    Args:
        model: GPT 模型（调用方负责 eval 与加载权重）。
        tok: 分词器。
        prompt: 用户问题。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度，<=0 时退化为 greedy。
        top_k: top-k 采样保留数，0 表示不启用。
        top_p: nucleus 采样阈值，1.0 表示不启用。
        repetition_penalty: 重复惩罚系数（对最近 32 个已生成 token 除以该系数）。
        stop_strings: 生成文本以其中任一字符串结尾时提前停止。
        min_tokens: 最小生成长度，之前屏蔽 EOS。
        device: 推理设备（torch.device 或 "cpu"/"cuda" 字符串）。
        wrap_prompt: True（默认）时把 prompt 视为裸问题，包装为 "用户:{prompt}\n助手:"；
            传入已拼接好的完整对话文本（多轮）时必须设为 False，避免双重包装。
        return_confidence: 是否返回逐 token 置信度。

    Returns:
        return_confidence 为 False 时返回生成文本；
        为 True 时返回 GenerateResult（文本 + 平均置信度 + 逐 token 置信度）。
    """
    model.eval()
    assert tok.bos_id is not None and tok.eos_id is not None, "分词器缺少 BOS/EOS 特殊 token"
    # normalize prompt: collapse or remove spaces commonly inserted in Chinese
    norm = prompt.replace(" ", "").replace("　", "")
    # wrap_prompt=False：prompt 已是完整对话文本（多轮拼接），不再包 "用户:\n助手:"
    text = "用户:" + norm + "\n助手:" if wrap_prompt else norm
    # 手动添加 BOS，不添加 EOS
    prefix = [tok.bos_id, *tok.encode(text, add_special_tokens=False)]
    x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)

    recent: list[int] = []
    kv_caches: list[KvCache | None] | None = None
    token_probs: list[float] = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # 推理优化：如果是第一步，输入全量 prefix；后续步仅输入上一个生成的 token + KV Cache
            cur_input = x if step == 0 else x[:, -1:]

            logits, kv_caches = model(cur_input, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(1e-6, temperature if temperature > 0 else 1.0)

            # 基础屏蔽逻辑：无论温度如何都生效
            if tok.pad_id is not None and tok.pad_id >= 0:
                logits[0, tok.pad_id] = -float("inf")
            if tok.bos_id is not None and tok.bos_id >= 0:
                logits[0, tok.bos_id] = -float("inf")
            if tok.unk_id is not None and tok.unk_id >= 0:
                logits[0, tok.unk_id] = -float("inf")

            # 强制最小生成长度，防止秒断
            if step < min_tokens and tok.eos_id is not None and tok.eos_id >= 0:
                logits[0, tok.eos_id] = -float("inf")

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
                token_probs.append(float(all_probs[0, int(next_id.item())].item()))

            x = torch.cat([x, next_id], dim=1)
            recent.append(int(next_id.item()))
            if next_id.item() == tok.eos_id:
                break
            if stop_strings:
                out_ids = x[0].tolist()[len(prefix) :]
                out_text = tok.decode(out_ids)
                if any(out_text.endswith(ss) for ss in stop_strings):
                    break
    out_ids = x[0].tolist()[len(prefix) :]
    text = _trim_leading_punct(tok.decode(out_ids))
    if return_confidence:
        avg_conf = sum(token_probs) / len(token_probs) if token_probs else 0.0
        return {"text": text, "avg_confidence": avg_conf, "token_probs": token_probs}
    return text


def main() -> None:
    """命令行推理入口。"""
    ap = argparse.ArgumentParser(description="命令行推理")
    ap.add_argument("--config", type=str, default="train/config.yml", help="配置文件路径")
    ap.add_argument("--ckpt", type=str, default="train/checkpoints/best.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.5)
    ap.add_argument("--stop_strings", nargs="*", default=None)
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
    packed = any("_packed_params" in k for k in sd)
    if packed:
        device = torch.device("cpu")
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)  # type: ignore[attr-defined]
    else:
        device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device("cpu")
    m.load_state_dict(sd)
    m.to(device)

    # 预热推理逻辑
    print("--- 模型推理结果 ---")
    text = generate(
        m,
        tok,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_strings=args.stop_strings,
        device=device,
    )
    if args.show_label:
        print("回答:" + text)
    else:
        print(text)


if __name__ == "__main__":
    main()
