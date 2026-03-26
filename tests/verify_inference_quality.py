import json

# 加载推理结果
with open("inference_test_results.json", "r", encoding="utf-8") as f:
    inference_results = json.load(f)

# 加载训练数据
train_data = {}
with open("data/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        train_data[obj["prompt"]] = obj["completion"]

print("=== 验证推理质量 ===")
print(f"{'序号':<4} {'状态':<8} {'正确率':<8} {'问题':<40}")
print("="*80)

for i, result in enumerate(inference_results, 1):
    question = result["question"]
    predicted = result["answer"]
    expected = train_data.get(question, "")

    if expected:
        # 计算匹配度
        if predicted == expected:
            match_ratio = 1.0
            status = "✓ 完全匹配"
        elif predicted in expected or expected.startswith(predicted):
            match_ratio = len(predicted) / len(expected)
            status = "○ 部分匹配"
        else:
            # 计算字符级别的相似度
            common_chars = sum(1 for c in predicted if c in expected)
            match_ratio = common_chars / max(len(predicted), len(expected)) if max(len(predicted), len(expected)) > 0 else 0
            status = "✗ 不匹配"

        print(f"{i:<4} {status:<8} {match_ratio:>6.1%} {question[:40]:<40}")

        if match_ratio < 0.8:
            print(f"  预测: {predicted[:80]}")
            print(f"  期望: {expected[:80]}")

print("\n=== 总结 ===")
perfect_matches = sum(1 for r in inference_results if train_data.get(r["question"]) == r["answer"])
partial_matches = sum(1 for r in inference_results if train_data.get(r["question"], "").startswith(r["answer"]))
print(f"完全匹配: {perfect_matches}/{len(inference_results)} ({perfect_matches/len(inference_results):.1%})")
print(f"部分匹配: {partial_matches}/{len(inference_results)} ({partial_matches/len(inference_results):.1%})")
