import os
import pickle
import random
import re
from openai import OpenAI
import json
from typing import List, Dict
from tqdm import tqdm

# ---------- 配置 ----------
MAX_ITER = 15  # 最多迭代轮数
with open("src/prompt/front.txt", "r", encoding="utf-8") as file:
    FRONT_PROMPT = file.read()
with open("src/prompt/back.txt", "r", encoding="utf-8") as file:
    BACK_PROMPT = file.read()
with open("src/prompt/middle.txt", "r", encoding="utf-8") as file:
    MIDDLE_PROMPT = file.read()

CLIENT = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-cb75705e482c4fd6afb1ec3f69ff06a4",
)
MODEL = "qwen-plus"
# --------------------------


def api_accuracy_score(predicted_apis: List[str], gold_apis: List[str]) -> float:
    if not gold_apis:
        return 0.0
    return len(set(predicted_apis).intersection(gold_apis)) / len(gold_apis)


def call_llm(prompt: str, question: Dict, related_apis: List[Dict]) -> List[str]:

    payload = json.dumps(
        {
            "mashup": question["mashup"],
            "candidate_apis": related_apis,
        },
        ensure_ascii=False,
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": payload},
    ]

    try:
        client = OpenAI(
            base_url="http://192.168.1.101:30111/v1",
            api_key="loopinnetwork",
        )
        response = client.chat.completions.create(
            model="Qwen2.5-14B-Instruct", messages=messages, stream=False
        )
        content = response.choices[0].message.content
        predicted_apis = process_json(content)
        return [api for api in predicted_apis if isinstance(api, str)][:10]
    except Exception as e:
        print("⚠️ LLM output parsing failed:", e)
        return []


def process_json(text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            json_data = json.loads(match.strip())  # 转换为 JSON 对象
            # 检查是否存在 "related_apis" 键
            if "related_apis" not in json_data:
                print("解析的 JSON 缺少 'related_apis' 键")
                return {"related_apis": []}
            return json_data["related_apis"]
        except json.JSONDecodeError as e:
            print("解析失败:", e)
            return []
        except Exception as e:
            print(f"没有找到JSON的开始或结束符: {e}")
            return []
    print("没有找到任何 JSON 代码块")
    return []


def select_representative_examples(full_dataset, all_outputs, k=10):
    # 选取得分高、中、低各部分样本
    indexed = list(zip(full_dataset, all_outputs))
    indexed.sort(key=lambda x: x[1]["score"], reverse=True)

    n = len(indexed)
    top = indexed[: k // 3]
    mid = indexed[n // 2 - k // 3 // 2 : n // 2 + k // 3 // 2]
    bottom = indexed[-k // 3 :]

    selected = top + mid + bottom
    random.shuffle(selected)  # 避免顺序偏置
    return selected


def reflect_and_improve_prompt(
    current_prompt: str,
    full_dataset: List[Dict],
    all_outputs: List[Dict],
    avg_score: float,
) -> str:
    system_msg = "You are an expert prompt engineer. Improve the prompt to maximize the accuracy of recommended APIs across all tasks."

    # ✅ 只用少量代表样本
    selected_examples = select_representative_examples(full_dataset, all_outputs, k=12)

    formatted_samples = ""
    for sample, output in selected_examples:
        apis = [api["title"] for api in sample["related_apis"]]
        formatted_samples += f"""
=== Task ===
Mashup: {sample['mashup']}
Gold APIs: {apis}
Predicted APIs: {output['predicted_apis']}
--- End Task ---
"""
    print(formatted_samples)
    user_input = f"""
Current Prompt:
{current_prompt}

Average Score: {avg_score * 100:.2f}%

Here are {len(selected_examples)} representative examples of the model's performance.
Please revise the prompt to improve API recommendation accuracy.

Only return the new improved prompt.
"""

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def run_global_prompt_optimization(dataset_path: str, save_path: str, resume=True):
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # 1️⃣ 初次筛选有效样本并采样
    filtered_samples = [s for s in all_samples if len(s.get("related_apis", [])) > 0]
    print(f"📊 {len(filtered_samples)} samples with related_apis > 0")

    if resume and os.path.exists(save_path):
        print("🧩 Resume mode: loading checkpoint...")
        with open(save_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
            samples = saved["samples"]
            prompt = saved["last_prompt"]
            middle = saved["last_middle"]
            history = saved.get("history", [])
            already_done = len(history)
    else:
        random.seed(42)  # ensure deterministic sampling
        samples = random.sample(filtered_samples, min(600, len(filtered_samples)))
        prompt = FRONT_PROMPT + "\n" + MIDDLE_PROMPT + "\n" + BACK_PROMPT
        middle = MIDDLE_PROMPT
        history = []
        already_done = 0

    print(f"✅ Using {len(samples)} training samples")
    print(f"🛠️ Already completed {already_done} iteration(s)")

    for iteration in range(already_done, MAX_ITER):
        print(f"\n🚀 Iteration {iteration + 1}/{MAX_ITER}")
        iteration_cache_path = f"{save_path}.iter{iteration}_cache.pkl"

        # ⏱️ 恢复当前 iteration 的处理缓存
        if os.path.exists(iteration_cache_path):
            print(f"🔁 Resuming from cache: {iteration_cache_path}")
            with open(iteration_cache_path, "rb") as f:
                cache = pickle.load(f)
                outputs = cache["outputs"]
                total_score = cache["total_score"]
                start_index = len(outputs)
        else:
            outputs = []
            total_score = 0.0
            start_index = 0

        # 样本循环
        for i in tqdm(range(start_index, len(samples)), desc=f"Evaluating prompt [iter {iteration+1}]"):
            sample = samples[i]
            try:
                related_apis = sample["related_apis"]
                gold_apis = sample["gold_apis"]

                predicted_apis = call_llm(prompt, sample, related_apis)
                score = api_accuracy_score(predicted_apis, gold_apis)

                outputs.append({"predicted_apis": predicted_apis, "score": score})
                total_score += score

            except Exception as e:
                print(f"⚠️ Error processing sample #{i}: {e}")
                outputs.append({"predicted_apis": [], "score": 0.0})

            # 每处理一条就保存缓存
            with open(iteration_cache_path, "wb") as f:
                pickle.dump({
                    "outputs": outputs,
                    "total_score": total_score
                }, f)

        # 处理完成后再统一计算平均分和 history
        avg_score = total_score / len(samples)
        history.append({
            "prompt": prompt,
            "avg_score": avg_score,
            "outputs": outputs
        })
        print(f"✅ Average Score: {avg_score:.4f}")

        # 保存完整中间状态（用于 resume）
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "samples": samples,
                "last_prompt": prompt,
                "last_score": avg_score,
                "last_middle": middle,
                "history": history
            }, f, ensure_ascii=False, indent=2)

        # 清除 iteration 缓存（可选）
        if os.path.exists(iteration_cache_path):
            os.remove(iteration_cache_path)

        # 反思优化 middle prompt
        middle = reflect_and_improve_prompt(middle, samples, outputs, avg_score)
        print(f"\n🧠 Updated Middle Prompt:\n{middle}")
        prompt = FRONT_PROMPT + "\n" + middle + "\n" + BACK_PROMPT

    # 最后挑选最优结果并保存
    best = max(history, key=lambda x: x["avg_score"])
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": samples,
            "best_prompt": best["prompt"],
            "best_score": best["avg_score"],
            "history": history
        }, f, ensure_ascii=False, indent=2)

    print(f"\n🏁 Optimization finished. Best score: {best['avg_score']:.4f}")
    print(f"✅ Best prompt saved to {save_path}")


run_global_prompt_optimization(
    dataset_path="data/prompt_train/seed_42/llm_generated_dataset.json",
    save_path="data/prompt_train/seed_42/best_middle_prompt.json",
    best_path="data/prompt_train/seed_42/best_final_prompt.json",
    resume=True,  # ✅ 默认启用断点续传
)
