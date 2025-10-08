import json
import random
from tqdm import tqdm

# ⚙️ 参数配置
POS_THRESHOLD = 0.6
NEG_THRESHOLD = 0
MAX_POS = 500
MAX_NEG = 500
SEED = 42

random.seed(SEED)


def extract_api_ids(mashup):
    return set(api["api_id"] for api in mashup.get("related_apis", []))


def compute_api_overlap(m1, m2):
    apis1 = extract_api_ids(m1)
    apis2 = extract_api_ids(m2)
    if not apis1 and not apis2:
        return 0.0
    return len(apis1 & apis2) / len(apis1 | apis2)


def build_flagembedding_data(mashups):
    samples = []
    n = len(mashups)

    for i in tqdm(range(n), desc="Building contrastive pairs"):
        anchor = mashups[i]
        positives = []
        negatives = []

        for j in range(n):
            if i == j:
                continue
            candidate = mashups[j]
            score = compute_api_overlap(anchor, candidate)

            if score >= POS_THRESHOLD:
                positives.append(candidate["description"])
            elif score <= NEG_THRESHOLD:
                negatives.append(candidate["description"])

        if positives and negatives:
            sample = {
                "query": anchor["description"],
                "pos": positives,
                "neg": negatives,
            }
            samples.append(sample)

    return samples


def save_to_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


# ======= 使用示例 =======

# 👇 请替换为你的 mashup 数据路径
input_path = "data/origin/seed42/train_data.json"  # 原始数据
output_path = "data/train_examples_" + str(POS_THRESHOLD) + ".jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    mashup_data = json.load(f)

dataset = build_flagembedding_data(mashup_data)
save_to_jsonl(dataset, output_path)

print(f"✅ 训练数据构建完成，已保存至: {output_path}")

input_path = "data/origin/seed42/val_data.json"  # 原始数据
output_path = "retrieval/data/val_examples_" + str(NEG_THRESHOLD) + ".jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    mashup_data = json.load(f)

dataset = build_flagembedding_data(mashup_data)
save_to_jsonl(dataset, output_path)

print(f"✅ 验证数据构建完成，已保存至: {output_path}")
