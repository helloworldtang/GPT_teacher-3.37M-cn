import gradio as gr
import torch
import time
import os
from src.model import GPT
from src.tokenizer import load_tokenizer
from src.infer import generate


def load_config(path):
    with open(path, "r") as f:
        import yaml
        return yaml.safe_load(f)


def load_model(ckpt_path, config_path):
    cfg = load_config(config_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cfg" in checkpoint:
        cfg = checkpoint["cfg"]

    tok = load_tokenizer(cfg["tokenizer"]["path"])

    model = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=0.0,
        n_kv_head=cfg["model"].get("n_kv_head"),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tok, cfg


DEFAULT_CKPT = "checkpoints/best.pt"
DEFAULT_CONFIG = "config/config.yml"

model = None
tokenizer = None
device = None


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


def chat(prompt, temperature, top_k, top_p):
    global model, tokenizer, device
    if model is None:
        if not os.path.exists(DEFAULT_CKPT):
            return "错误：未找到模型文件 checkpoints/best.pt，请先运行训练。\n\n运行命令: uv run python run.py"
        model, tokenizer, _ = load_model(DEFAULT_CKPT, DEFAULT_CONFIG)
        device = get_device()
        model.to(device)

    start = time.time()
    response = generate(
        model, tokenizer, prompt,
        max_new_tokens=128,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.5,
        stop_strings=["用户:", "\n用户", "。"],
        device=device,
    )
    elapsed = time.time() - start
    return f"{response}\n\n---\n推理耗时: {elapsed:.2f}s"


# 快捷问题
EXAMPLE_QUESTIONS = [
    "什么是注意力机制？",
    "解释一下 RoPE 的作用",
    "RMSNorm 和 LayerNorm 有什么区别？",
    "计算 15 乘以 6 是多少？",
    "你好，你是谁？",
    "太阳系有哪些行星？",
    "蒸馏水和纯水有什么区别？",
    "什么是开源软件？",
]

with gr.Blocks(
    title="GPT Teacher 教学演示",
    css="""
        .example-btn { min-width: 120px; margin: 4px; }
        .param-info { font-size: 12px; color: #666; }
    """
) as demo:

    gr.Markdown(
        "# GPT Teacher 教学演示\n"
        "这是一个 ~3.37M 参数的微型 GPT 模型，展示了 Transformer Decoder-only 架构的推理过程。\n\n"
        "**快速开始**: 点击下方问题按钮，或输入自己的问题。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            input_box = gr.Textbox(
                label="输入问题",
                placeholder="输入你想问的问题...",
                lines=2,
            )
            output_box = gr.Textbox(
                label="模型回答",
                lines=6,
            )

            with gr.Row():
                submit_btn = gr.Button("提问", variant="primary")
                clear_btn = gr.Button("清空")

        with gr.Column(scale=1):
            temperature = gr.Slider(
                0.0, 1.0, value=0.0, step=0.1,
                label="Temperature (温度)",
                info="0=精确复制训练答案，1=更有创造性",
            )
            top_k = gr.Slider(
                1, 100, value=50, step=1,
                label="Top-K",
                info="只从概率最高的K个词中选，越小越确定",
            )
            top_p = gr.Slider(
                0.0, 1.0, value=0.9, step=0.05,
                label="Top-P (核采样)",
                info="累积概率阈值，越小越保守",
            )

    gr.Markdown("### 点击试试这些问题")
    with gr.Row():
        for q in EXAMPLE_QUESTIONS[:4]:
            gr.Button(q, elem_classes="example-btn").click(
                lambda x=q: (x, chat(x, 0.0, 50, 0.9)),
                inputs=[], outputs=[input_box, output_box],
            )
    with gr.Row():
        for q in EXAMPLE_QUESTIONS[4:]:
            gr.Button(q, elem_classes="example-btn").click(
                lambda x=q: (x, chat(x, 0.0, 50, 0.9)),
                inputs=[], outputs=[input_box, output_box],
            )

    submit_btn.click(chat, [input_box, temperature, top_k, top_p], output_box)
    input_box.submit(chat, [input_box, temperature, top_k, top_p], output_box)
    clear_btn.click(lambda: ("", ""), inputs=[], outputs=[input_box, output_box])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
    )
