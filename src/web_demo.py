import gradio as gr
import torch
import time
import os
from src.model import GPT
from src.tokenizer import load_tokenizer
from src.infer import generate



def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]

    tok = load_tokenizer(
        cfg.get("tokenizer", {}).get("type", "byte"),
        cfg.get("tokenizer", {}).get("path"),
    )

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

model = None
tokenizer = None
device = None
model_info = ""


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


def chat(prompt, temperature, top_k, top_p, max_tokens, repeat_penalty):
    global model, tokenizer, device, model_info
    if model is None:
        if not os.path.exists(DEFAULT_CKPT):
            return "错误：未找到模型文件 checkpoints/best.pt，请先运行训练。\n\n运行命令: uv run python -m src.train"
        model, tokenizer, cfg = load_model(DEFAULT_CKPT)
        device = get_device()
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        model_info = f"模型: {n_params/1e6:.2f}M 参数 | 设备: {device}"

    start = time.time()
    response = generate(
        model, tokenizer, prompt,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        repetition_penalty=repeat_penalty,
        stop_strings=["用户:", "\n用户", "。"],
        device=device,
    )
    elapsed = time.time() - start
    return f"{response}\n\n---\n推理耗时: {elapsed:.2f}s"


# 快捷问题
EXAMPLE_QUESTIONS = [
    "什么是注意力机制？",
    "RoPE 是什么？",
    "什么是机器学习？",
    "计算 15 乘以 6 是多少？",
    "你是谁？",
    "太阳系有哪些行星？",
    "蒸馏水和纯水有什么区别？",
    "权重共享有什么好处？",
]

with gr.Blocks(
    title="GPT Teacher 教学演示",
    css="""
        .example-btn { min-width: 120px; margin: 4px; }
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
                0.0, 1.5, value=0.0, step=0.1,
                label="Temperature (温度)",
                info="0=精确复制，1=有创造性",
            )
            top_k = gr.Slider(
                1, 100, value=50, step=1,
                label="Top-K",
                info="只从概率最高的K个词中选",
            )
            top_p = gr.Slider(
                0.0, 1.0, value=0.9, step=0.05,
                label="Top-P (核采样)",
                info="累积概率阈值",
            )
            with gr.Accordion("高级参数", open=False):
                max_tokens = gr.Slider(
                    16, 256, value=128, step=16,
                    label="Max Tokens (最大生成长度)",
                    info="控制回答的最大长度",
                )
                repeat_penalty = gr.Slider(
                    1.0, 2.0, value=1.5, step=0.1,
                    label="Repetition Penalty (重复惩罚)",
                    info="防止模型重复说同样的话",
                )
            model_info_box = gr.Markdown(
                "模型尚未加载，点击「提问」自动加载"
            )

    gr.Markdown("### 点击试试这些问题")
    with gr.Row():
        for q in EXAMPLE_QUESTIONS[:4]:
            gr.Button(q, elem_classes="example-btn").click(
                lambda x=q: (x, chat(x, 0.0, 50, 0.9, 128, 1.5)),
                inputs=[], outputs=[input_box, output_box],
            )
    with gr.Row():
        for q in EXAMPLE_QUESTIONS[4:]:
            gr.Button(q, elem_classes="example-btn").click(
                lambda x=q: (x, chat(x, 0.0, 50, 0.9, 128, 1.5)),
                inputs=[], outputs=[input_box, output_box],
            )

    def chat_and_show_info(prompt, temperature, top_k, top_p, max_tokens, repeat_penalty):
        result = chat(prompt, temperature, top_k, top_p, max_tokens, repeat_penalty)
        return result, model_info

    submit_btn.click(
        chat_and_show_info,
        [input_box, temperature, top_k, top_p, max_tokens, repeat_penalty],
        [output_box, model_info_box],
    )
    input_box.submit(
        chat_and_show_info,
        [input_box, temperature, top_k, top_p, max_tokens, repeat_penalty],
        [output_box, model_info_box],
    )
    clear_btn.click(
        lambda: ("", "", model_info or "模型尚未加载，点击「提问」自动加载"),
        inputs=[], outputs=[input_box, output_box, model_info_box],
    )


if __name__ == "__main__":
    import os
    print("\n" + "=" * 50)
    print("  GPT Teacher Web Demo 启动成功！")
    print("  打开浏览器访问: http://127.0.0.1:7860")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50 + "\n")
    demo.queue().launch(
        server_name=os.environ.get("SERVER_NAME", "127.0.0.1"),
        server_port=7860,
        show_error=True,
        share=False,
    )
