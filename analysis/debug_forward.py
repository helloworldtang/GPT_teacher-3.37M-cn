import torch
import torch.nn.functional as F
from src.model import GPT
import yaml
from tokenizers import Tokenizer

# 加载配置
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 加载分词器
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint = torch.load("checkpoints/last.pt", map_location=device)
vocab_size = checkpoint["model"]["tok_emb.weight"].shape[0]

model = GPT(
    vocab_size=vocab_size,
    n_layer=config["model"]["n_layer"],
    n_head=config["model"]["n_head"],
    n_embd=config["model"]["n_embd"],
    seq_len=config["model"]["seq_len"],
    dropout=config["model"]["dropout"],
    use_flash=True,
)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

def test_forward(prompt):
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}")

    # 使用与推理相同的格式
    full_prompt = "用户:" + prompt + "\n助手:"
    print(f"Full prompt: {full_prompt}")

    # Tokenize
    tokens = tokenizer.encode(full_prompt).ids
    print(f"Tokens: {tokens}")
    print(f"Tokens text: {tokenizer.decode(tokens)}")

    # 转换为tensor
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    print(f"Input shape: {x.shape}")

    # 前向传播
    with torch.no_grad():
        logits = model(x)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # 获取最后一个token的logits
    next_token_logits = logits[0, -1, :]
    print(f"\nNext token logits shape: {next_token_logits.shape}")
    print(f"Next token logits range: [{next_token_logits.min():.2f}, {next_token_logits.max():.2f}]")

    # 获取top 10 tokens
    top_k = 10
    top_logits, top_indices = torch.topk(next_token_logits, top_k)
    print(f"\nTop {top_k} next tokens:")
    for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
        token_text = tokenizer.decode([idx.item()])
        print(f"  {i+1}. Token {idx.item()} (logit={logit:.2f}, prob={F.softmax(next_token_logits, dim=-1)[idx]:.4f}): '{token_text}'")

    # 检查训练数据中的第一个token
    import json
    with open("data/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    for i, sample in enumerate(train_data):
        if prompt in sample["prompt"]:
            completion = sample["completion"]
            print(f"\n训练数据中的completion: {completion}")

            # 编码完整的训练序列
            train_full = "用户:" + sample["prompt"] + "\n助手:" + completion
            train_tokens = tokenizer.encode(train_full).ids
            print(f"训练完整序列tokens: {train_tokens}")

            # 找到"\n助手:"之后的首个token
            assistant_prefix = tokenizer.encode("\n助手:").ids
            print(f"\\n助手: tokens: {assistant_prefix}")

            # 找到assistant之后的第一个token
            if len(tokens) < len(train_tokens):
                first_completion_token = train_tokens[len(tokens)]
                print(f"\n训练数据中assistant后的首个token: {first_completion_token}")
                token_text = tokenizer.decode([first_completion_token])
                print(f"Token text: '{token_text}'")
                print(f"该token的logit: {next_token_logits[first_completion_token].item():.2f}")
                print(f"该token的概率: {F.softmax(next_token_logits, dim=-1)[first_completion_token].item():.4f}")
                print(f"该token在top {top_k}中的排名: {(next_token_logits.argsort(descending=True) == first_completion_token).nonzero().item() + 1}")

                # 检查top tokens中是否有completion的token
                completion_tokens = tokenizer.encode(completion).ids
                print(f"\nCompletion tokens: {completion_tokens}")
                print(f"Completion tokens在top {top_k}中:")
                for ct in completion_tokens[:5]:
                    rank = (next_token_logits.argsort(descending=True) == ct).nonzero(as_tuple=True)[0]
                    if len(rank) > 0:
                        print(f"  Token {ct}: 排名{rank.item()+1}, logit={next_token_logits[ct]:.2f}, prob={F.softmax(next_token_logits, dim=-1)[ct]:.4f}")
                    else:
                        print(f"  Token {ct}: 未找到")
            break

    return next_token_logits

# 测试三个问题
print("\n" + "="*80)
print("测试1: 蒸馏水与纯水区别（应该能正确回答）")
print("="*80)
logits1 = test_forward("解释蒸馏水与纯水区别？")

print("\n" + "="*80)
print("测试2: 如何计算1+2+3+...+100的和（输出错误）")
print("="*80)
logits2 = test_forward("如何计算1+2+3+...+100的和？")

print("\n" + "="*80)
print("测试3: 什么是质数（输出为空）")
print("="*80)
logits3 = test_forward("什么是质数？")

# 比较logits分布
print("\n" + "="*80)
print("Logits分布比较")
print("="*80)
for i, logits in enumerate([logits1, logits2, logits3], 1):
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_prob = probs.max().item()
    print(f"\n问题{i}:")
    print(f"  熵: {entropy:.4f} (越高越不确定)")
    print(f"  最大概率: {max_prob:.4f}")
    print(f"  Top 1 token: {torch.argmax(logits).item()} ('{tokenizer.decode([torch.argmax(logits).item()])}')")
