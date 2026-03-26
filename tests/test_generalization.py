import subprocess

# 测试一些不在训练集中的问题
test_questions = [
    "什么是深度学习？",
    "解释一下机器学习的基本概念",
    "什么是神经网络？",
    "什么是梯度下降？",
    "什么是损失函数？",
    "什么是过拟合？",
    "什么是反向传播？",
]

print("=== 测试模型泛化能力 ===")
print(f"{'序号':<4} {'问题':<50}")
print("="*80)

for i, question in enumerate(test_questions, 1):
    print(f"\n[{i}/{len(test_questions)}] {question}")

    try:
        result = subprocess.run(
            ["python", "-m", "src.infer", "--prompt", question],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # 提取输出
            lines = result.stdout.strip().split('\n')
            answer_lines = []
            for line in lines:
                if line.strip() and not any(x in line for x in ['FlashAttention', '层数:', '头数:', '嵌入', '序列', 'Flash', '===']):
                    answer_lines.append(line.strip())

            answer = '\n'.join(answer_lines) if answer_lines else ""
            print(f"回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        else:
            print(f"错误: {result.stderr}")
    except Exception as e:
        print(f"异常: {e}")

print("\n=== 泛化测试完成 ===")
