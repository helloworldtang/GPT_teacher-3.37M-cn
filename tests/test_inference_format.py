import torch
import torch.nn.functional as F
from src.model import GPT
from src.tokenizer import load_tokenizer
import yaml

# 加载配置
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 加载分词器
tokenizer = load_tokenizer(
    config.get("tokenizer", {}).get("type", "byte"),
    config.get("tokenizer", {}).get("path"),
)

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("checkpoints/last.pt", map_location=device)

model = GPT(
    vocab_size=tokenizer.vocab_size,
    n_layer=config["model"]["n_layer"],
    n_head=config["model"]["n_head"],
    n_embd=config["model"]["n_embd"],
    seq_len=config["model"]["seq_len"],
    dropout=config["model"]["dropout"],
    use_flash=False,
)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

def test_inference(prompt, format_type="current"):
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Format: {format_type}")
    print(f"{'='*80}")

    if format_type == "current":
        # 当前推理格式
        full_prompt = "用户:" + prompt + "\n助手:"
    elif format_type == "with_bos":
        # 加上BOS的格式（匹配训练）
        full_prompt = "用户:" + prompt + "\n助手:" + tokenizer.decode([tokenizer.bos_id])
    else:
        # 简化格式
        full_prompt = prompt

    tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
    print(f"Full prompt: {full_prompt}")
    print(f"Tokens: {tokens}")

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        next_token_logits = logits[0, -1, :]

    # 获取top 10 tokens
    top_k = 10
    top_logits, top_indices = torch.topk(next_token_logits, top_k)
    print(f"\nTop {top_k} next tokens:")
    for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
        token_text = tokenizer.decode([idx.item()])
        print(f"  {i+1}. Token {idx.item()} (logit={logit:.2f}, prob={F.softmax(next_token_logits, dim=-1)[idx]:.4f}): '{token_text}'")

    # 生成前几个token
    generated_tokens = []
    with torch.no_grad():
        for _ in range(10):
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0)
            generated_tokens.append(next_id.item())
            x = torch.cat([x, next_id], dim=1)

    generated_text = tokenizer.decode(generated_tokens)
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Generated text: '{generated_text}'")

# 测试三个问题
test_inference("解释蒸馏水与纯水区别？", "current")
test_inference("解释蒸馏水与纯水区别？", "with_bos")
test_inference("如何计算1+2+3+...+100的和？", "current")
test_inference("如何计算1+2+3+...+100的和？", "with_bos")
test_inference("什么是质数？", "current")
test_inference("什么是质数？", "with_bos")
