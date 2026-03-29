import torch
from src.model import GPT
from src.tokenizer import load_tokenizer

tok = load_tokenizer('hf_tokenizers', 'tokenizer/tokenizer.json')
obj = torch.load('checkpoints/last.pt', map_location='cpu')
cfg = obj['cfg']

m = GPT(
    vocab_size=tok.vocab_size,
    n_layer=cfg['model']['n_layer'],
    n_head=cfg['model']['n_head'],
    n_embd=cfg['model']['n_embd'],
    seq_len=cfg['model']['seq_len'],
    dropout=cfg['model']['dropout'],
    use_flash=False,
)
m.load_state_dict(obj['model'])
m.eval()

prompts = [
    ('什么是位置编码？', '位置编码'),
    ('为什么需要注意力？', '关注重要'),
]

for prompt, keyword in prompts:
    prefix = tok.encode('用户:' + prompt + '\n助手:', add_special_tokens=True)
    x = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        logits = m(x)
        probs = torch.softmax(logits[0, -1], dim=-1)
        top5 = torch.topk(probs, 5)
        
        print(f'\nQ: {prompt}')
        print(f'Expected keyword: {keyword}')
        print('Top 5 predictions:')
        for i, (p, idx) in enumerate(zip(top5.values, top5.indices)):
            tid = idx.item()
            print(f'  {i+1}. [{p.item()*100:.1f}%] {repr(tok.decode([tid])[:40])}')
