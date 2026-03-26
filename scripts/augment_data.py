import json
import random
from typing import List, Dict
import argparse

class DataAugmenter:
    def __init__(self, seed=42):
        random.seed(seed)
        
        self.question_templates = {
            "what": ["什么是", "什么是", "解释一下"],
            "how": ["如何", "怎么", "怎样"],
            "why": ["为什么", "为何", "原因是什么"],
            "when": ["什么时候", "何时"],
            "where": ["在哪里", "何地"],
            "who": ["谁", "哪个人"],
        }
        
        self.answer_variants = {
            "是": ["就是", "也就是", "表示"],
            "可以": ["能够", "可以"],
            "需要": ["要", "必须"],
            "因为": ["由于", "因为"],
            "所以": ["因此", "所以"],
        }
    
    def augment_question(self, question: str) -> str:
        if "什么是" in question:
            return question.replace("什么是", random.choice(self.question_templates["what"]))
        elif "如何" in question:
            return question.replace("如何", random.choice(self.question_templates["how"]))
        elif "为什么" in question:
            return question.replace("为什么", random.choice(self.question_templates["why"]))
        elif "什么时候" in question:
            return question.replace("什么时候", random.choice(self.question_templates["when"]))
        elif "在哪里" in question:
            return question.replace("在哪里", random.choice(self.question_templates["where"]))
        elif "谁" in question:
            return question.replace("谁", random.choice(self.question_templates["who"]))
        else:
            return question
    
    def augment_answer(self, answer: str) -> str:
        augmented = answer
        for original, variants in self.answer_variants.items():
            if original in augmented:
                augmented = augmented.replace(original, random.choice(variants))
        return augmented
    
    def create_answer_variant(self, answer: str) -> str:
        sentences = answer.split("。")
        if len(sentences) > 1:
            random.shuffle(sentences[:-1])
            return "。".join(sentences)
        return answer
    
    def augment_sample(self, sample: Dict) -> List[Dict]:
        augmented_samples = [sample]
        
        augmented_question = self.augment_question(sample["prompt"])
        if augmented_question != sample["prompt"]:
            augmented_samples.append({
                "prompt": augmented_question,
                "completion": sample["completion"]
            })
        
        augmented_answer = self.augment_answer(sample["completion"])
        if augmented_answer != sample["completion"]:
            augmented_samples.append({
                "prompt": sample["prompt"],
                "completion": augmented_answer
            })
        
        variant_answer = self.create_answer_variant(sample["completion"])
        if variant_answer != sample["completion"]:
            augmented_samples.append({
                "prompt": sample["prompt"],
                "completion": variant_answer
            })
        
        return augmented_samples

def augment_dataset(input_path: str, output_path: str, augment_factor: int = 2):
    augmenter = DataAugmenter()
    
    original_samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            original_samples.append(json.loads(line))
    
    augmented_samples = []
    for sample in original_samples:
        augmented = augmenter.augment_sample(sample)
        augmented_samples.extend(augmented[:augment_factor + 1])
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"=== 数据增强完成 ===")
    print(f"原始样本数: {len(original_samples)}")
    print(f"增强后样本数: {len(augmented_samples)}")
    print(f"增强倍数: {len(augmented_samples) / len(original_samples):.2f}x")
    print(f"输出文件: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/train.jsonl")
    parser.add_argument("--output", type=str, default="data/train_augmented.jsonl")
    parser.add_argument("--augment_factor", type=int, default=2, help="每个样本生成多少个增强版本")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    augment_dataset(args.input, args.output, args.augment_factor)

if __name__ == "__main__":
    main()
