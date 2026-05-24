# GPT Teacher — Train a Small Chinese GPT from Scratch on CPU

## Why This Project

Most ML tutorials teach theory but skip the hardest part: making the model actually work. Students follow along, loss drops, everything looks great — until they ask the model a question and get gibberish.

This project's goal: **let beginners run through a complete GPT training pipeline and get usable results.** From tokenization to training to evaluation, every step has a clear "did it work?" checkpoint.

- **One command to run everything**: tokenizer → train → evaluate → web demo
- **Automatic evaluation**: see which questions the model learned to answer
- **Loss curve visualization**: watch training converge in real time
- **Interactive web demo**: try different prompts and parameters

## Who Is This For

- Beginners who want to get into ML but don't know where to start
- Students who've studied theory but never run a complete project
- Developers who want to quickly understand the core GPT pipeline

## What You'll Learn

- GPT core principles:
  - Self-attention (capturing relationships between tokens)
  - Causal masking (ensuring the model only sees past tokens)
  - Transformer architecture (efficient sequence modeling)
- Full pipeline: embedding → positional encoding → attention → feed-forward → residual → normalization → causal mask → loss → backprop → optimize → save → inference
- Efficient small-model techniques: RMSNorm, RoPE, GQA, SwiGLU, weight tying, short sequences, small vocabulary, quantization, early stopping
- CPU-only training: gradient accumulation, learning rate warmup & cosine annealing, disabling unnecessary features

## Quick Start

Requires **Python 3.10-3.12** (PyTorch does not support 3.13+).

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11 and sync dependencies
uv python install 3.11
uv sync

# Run the full pipeline: build tokenizer → train → evaluate → web demo
uv run python run.py
```

**Expected result**: `Result: 6/6 passed (100%)` — the model learned to answer all test questions.

## Training Results

- **Model**: 3.16M parameters, Llama-style architecture (GQA + SwiGLU + RMSNorm + RoPE)
- **Training time**: ~5-6 min on Apple Silicon (MPS), ~30-60 min on CPU
- **Evaluation**: 6/6 (100%) pass rate

### Example Outputs

```
Q: What is attention?
A: Attention computes relevance between queries and keys to assign weights, letting the model focus on the most relevant parts.

Q: What is RoPE?
A: RoPE is Rotary Position Embedding, injecting relative position information through complex rotation.

Q: What's the difference between distilled water and purified water?
A: Distilled water is produced through distillation and condensation, purified water has very low impurities.
```

## Project Structure

- `src/model.py` — GPT core (GQA attention / SwiGLU feed-forward / RMSNorm / RoPE / weight tying)
- `src/data.py` — Tokenizer loading, instruction data processing, batching
- `src/train.py` — Training loop with evaluation, checkpointing, and loss visualization
- `src/infer.py` — Inference CLI (temperature, top-k, top-p, repetition penalty, stop strings)
- `src/evaluate.py` — Evaluation script (runs test.jsonl, reports pass/fail)
- `src/visualize.py` — Concept visualizations (causal mask, real attention weights, tokenizer, loss curves)
- `src/tokenizer.py` / `src/build_tokenizer.py` — BPE tokenizer loading & building
- `data/prepare_data.py` — Training data generation with augmentation
- `config/config.yml` — Centralized configuration
- `run.py` — One-click launcher
- `notebooks/tutorial.ipynb` — Step-by-step Jupyter tutorial

## Architecture

```
Input → Tokenizer → Token IDs → Embedding →
  ×4 [RMSNorm → GQA Attention + Residual → RMSNorm → SwiGLU + Residual] →
RMSNorm → LM Head (weight tied) → Next token probabilities
```

### Configuration (config/config.yml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_layer | 4 | Transformer layers |
| n_head | 4 | Attention heads |
| n_kv_head | 2 | KV heads (GQA) |
| n_embd | 256 | Embedding dimension |
| seq_len | 128 | Maximum sequence length |
| vocab_size | ~1500 | BPE vocabulary size |
| **Total params** | **~3.16M** | CPU-friendly |

### Training

- Optimizer: AdamW, lr=0.001, weight_decay=0.1
- Schedule: Linear warmup (100 steps) + cosine annealing
- Batch: micro_batch=4, batch_size=16 (gradient accumulation)
- Early stopping: patience=15 evaluations
- Max steps: 5000

## Device Support

| Device | Speed | Notes |
|--------|-------|-------|
| MPS (Apple Silicon) | ~5-6 min | Recommended for Mac users |
| CUDA (NVIDIA GPU) | ~2-3 min | Fastest option |
| CPU | ~30-60 min | Works everywhere |

## Visualizations

```bash
# Generate all visualization charts
uv run python -m src.visualize

# Real attention weights from trained model
uv run python -m src.visualize --only real_attention --prompt "什么是注意力机制？"

# BPE tokenizer visualization
uv run python -m src.visualize --only tokenizer --prompt "注意力机制通过计算相关性分配权重"
```

## Experiments

### Beginner

1. Try different temperatures (0, 0.5, 1.0) on the same question
2. Change `max_steps` to 100 and retrain — what happens?

### Intermediate

3. Add a new Q&A pair in `data/prepare_data.py`, regenerate data, and retrain
4. Change `n_layer` from 4 to 2 — can a smaller model learn the same knowledge?

### Advanced

5. Change `seq_len` from 128 to 64 — which answers get worse?
6. Increase `n_embd` from 256 to 384 — observe the capacity change

## Troubleshooting

- **uv sync fails with torch**: Run `uv python install 3.11` and retry
- **Garbled output**: Ensure `src/build_tokenizer.py` was run with ByteLevel decoder
- **Empty output**: Increase training steps (≥1500) and check stop string settings
- **MPS fails**: Check `torch.backends.mps.is_available()` returns True

## License

MIT

---

Project: https://github.com/helloworldtang/GPT_teacher-3.37M-cn
