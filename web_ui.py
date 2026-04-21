import os
import torch
import gradio as gr

# 从项目的 src 目录导入已有的核心方法
from src.infer import load_checkpoint, generate
from src.tokenizer import load_tokenizer
from src.model import GPT

# ==========================================
# 1. 全局初始化：加载模型与分词器 (只加载一次)
# ==========================================
print("正在加载模型，请稍候...")
CKPT_PATH = "checkpoints/last.pt"  # 默认使用最后一次训练的权重
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

try:
    obj = load_checkpoint(CKPT_PATH)
    cfg = obj["cfg"]
    tok = load_tokenizer(cfg.get("tokenizer", {}).get("type", "byte"), cfg.get("tokenizer", {}).get("path"))
    
    # 实例化模型
    model = GPT(
        vocab_size=tok.vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        seq_len=cfg["model"]["seq_len"],
        dropout=cfg["model"]["dropout"],
        use_flash=False
    )
    model.load_state_dict(obj["model"])
    model.to(DEVICE)
    model.eval()
    print(f"模型加载成功！当前使用设备: {DEVICE}")
except Exception as e:
    print(f"模型加载失败，请检查权重文件是否存在。错误信息: {e}")
    exit(1)

# ==========================================
# 2. 定义对话响应函数
# ==========================================
def chat_predict(message, history, temperature, top_p, max_new_tokens, repetition_penalty):
    """
    Gradio 的 ChatInterface 会自动传入 message (当前输入) 和 history (历史记录)。
    后面的参数是我们自定义的控制滑块。
    """
    # 提示词规范化与停止词设置（复用 infer.py 的优秀策略）
    stop_strings = ["。", "；", "\n"] 
    
    # 调用现有的生成逻辑
    response = generate(
        model=model,
        tok=tok,
        prompt=message,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
        device=torch.device(DEVICE)
    )
    return response

# ==========================================
# 3. 构建并启动 Gradio 界面
# ==========================================
# 定义侧边栏的高级控制参数
additional_inputs = [
    gr.Slider(minimum=0.0, maximum=2.0, value=0.8, step=0.1, label="Temperature (温度 - 控制创造力)"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P (核采样 - 控制词汇多样性)"),
    gr.Slider(minimum=10, maximum=256, value=64, step=1, label="Max Tokens (最大生成长度)"),
    gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition Penalty (重复惩罚)"),
]

# 创建对话界面
demo = gr.ChatInterface(
    fn=chat_predict,
    title="🤖 GPT Teacher (3.37M 中文微模型)",
    description="这是一个在 CPU 上训练的小型中文 GPT 模型。由于模型极小，它擅长回答训练集内出现过的知识点。",
    examples=[
        ["什么是注意力机制？"], 
        ["解释蒸馏水与纯水区别？"], 
        ["深度学习有什么用？"]
    ],
    additional_inputs=additional_inputs,
    # 删掉了 theme, submit_btn, retry_btn, undo_btn, clear_btn
)

if __name__ == "__main__":
    # 启动 Web 服务
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)