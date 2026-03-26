import subprocess
import json

# 从训练数据中提取所有问题
questions = []
with open("data/train.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        questions.append(obj["prompt"])

# 测试前10个问题
print("=== 批量测试推理 ===")
results = []
for i, question in enumerate(questions[:10]):
    print(f"\n[{i+1}/10] Question: {question}")

    try:
        result = subprocess.run(
            ["python", "-m", "src.infer", "--prompt", question],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # 提取输出（去除配置信息）
            lines = result.stdout.strip().split('\n')
            answer_lines = []
            in_answer = False
            for line in lines:
                if 'Flash Attention' in line or '===' in line or '===' in line or '层数' in line or '头数' in line or '嵌入' in line or '序列' in line:
                    if in_answer and line.strip():
                        answer_lines.append(line.strip())
                    continue
                if '===' in line:
                    continue
                if line.strip() and not any(x in line for x in ['FlashAttention', '层数:', '头数:', '嵌入', '序列', 'Flash']):
                    in_answer = True
                    answer_lines.append(line.strip())

            answer = '\n'.join(answer_lines) if answer_lines else "No answer"
            print(f"Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            results.append({"question": question, "answer": answer, "status": "success"})
        else:
            print(f"Error: {result.stderr}")
            results.append({"question": question, "answer": "", "status": "error"})
    except subprocess.TimeoutExpired:
        print("Timeout")
        results.append({"question": question, "answer": "", "status": "timeout"})
    except Exception as e:
        print(f"Exception: {e}")
        results.append({"question": question, "answer": "", "status": "exception"})

# 保存结果
with open("inference_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n\n=== 测试完成 ===")
print(f"成功: {sum(1 for r in results if r['status'] == 'success')}/10")
print(f"失败: {sum(1 for r in results if r['status'] != 'success')}/10")
