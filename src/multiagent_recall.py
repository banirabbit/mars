# src/multiagent_recommender.py

import ast
import json
import re
import time
from typing import Dict, List, Tuple, TypedDict
from langgraph.graph import StateGraph, END
import os
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

# Default number of APIs to recommend in prompt/output
DEFAULT_TOP_K = 10

# Review prompt for picking the best option
REVIEW_PROMPT = """You are a strict reviewer selecting the best API combination.

Rules:
- Compare the options RELATIVELY.
- Do NOT count or compare quantities.
- Do NOT propose new APIs.
- Do NOT suggest improvements.
- Judge by combined semantic coverage of the mashup requirements.
- Choose the best achievable option within the candidate set.

Output ONLY JSON:
{"best_index": <int>, "confidence": <0..1>}
"""

# Global variables for OpenAI client and model - will be initialized by setup_llm_client()
CLIENT = None
MODEL = None
MAX_RETRY_COUNT = 5


def setup_llm_client(base_url, api_key, model_name, max_retry_count=5):
    """Setup OpenAI client and model from configuration."""
    global CLIENT, MODEL, MAX_RETRY_COUNT
    CLIENT = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    MODEL = model_name
    MAX_RETRY_COUNT = max_retry_count


def _extract_usage(completion):
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (
        prompt_tokens + completion_tokens
    )
    return prompt_tokens, completion_tokens, total_tokens


def select_top_apis(state):
    """
    一次性生成多个候选方案（M=3），不 retry
    """
    apis = state["candidate_apis"]
    mashups = state["mashup"]
    core_prompt = state["prompt"]

    prompt_payload = {"mashup": mashups, "candidate_apis": apis}
    user_prompt = json.dumps(prompt_payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": core_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if CLIENT is None or MODEL is None:
        raise RuntimeError("LLM client not initialized. Call setup_llm_client() first.")

    options = []
    call_logs = list(state.get("call_logs", []))
    total_duration = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_sum = 0

    # 生成 3 个不同的方案
    for i in range(3):
        start_time = time.time()
        completion = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,  # 保证多样性
            stream=False
        )
        duration = time.time() - start_time
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(completion)

        response = completion.choices[0].message.content
        print("options:")
        print(f"[Option {i}]:", response)

        options.append(process_json(response))

        total_duration += duration
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens_sum += total_tokens

    call_logs.append(
        {
            "stage": "select_top_apis",
            "num_options": 3,
            "duration_sec": total_duration,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens_sum,
        }
    )

    return {
        **state,
        "related_apis_options": options,  # 存储多个方案
        "call_logs": call_logs,
        "total_time": state.get("total_time", 0.0) + total_duration,
        "total_tokens": state.get("total_tokens", 0) + total_tokens_sum,
        "iterations": state.get("iterations", 0) + 1,
    }


def process_json(text: str):
    """
    在文本中查找 JSON（支持 ```json ... ``` 代码块和内联 { ... } 结构），
    解析并返回其中所有出现的 related_apis（去重后按出现顺序返回）。
    向下兼容：如果只存在代码块里的 JSON，行为等同于原先版本。
    """
    candidates = []

    # 1) 先找 ```json ... ``` 代码块（保留原有功能）
    fenced_pattern = r"```json\s*(.*?)\s*```"
    fenced_matches = re.findall(fenced_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend([m.strip() for m in fenced_matches])

    # 2) 再扫描内联 { ... }，用括号匹配找出可能的 JSON 子串
    #    只收录包含 related_apis 关键字的子串，减少误伤
    for i, ch in enumerate(text):
        if ch == '{':
            stack = 1
            j = i + 1
            while j < len(text) and stack > 0:
                if text[j] == '{':
                    stack += 1
                elif text[j] == '}':
                    stack -= 1
                j += 1
            if stack == 0:
                snippet = text[i:j]
                if "related_apis" in snippet:
                    candidates.append(snippet)

    # 去重（按文本片段）
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    results = []
    def try_parse(candidate: str):
        """尽量把 candidate 解析成 dict；做一些常见容错。"""
        # 直接尝试 json.loads
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 尝试修正：中英文引号、尾逗号
        fixed = (candidate
                 .replace('“', '"').replace('”', '"')
                 .replace('′', "'").replace('’', "'"))
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except Exception:
            pass

        # 最后尝试 ast.literal_eval（对 “类 JSON” 也更宽容）
        try:
            return ast.literal_eval(candidate)
        except Exception as e:
            print("解析失败:", e)
            return None

    # 解析候选并收集 related_apis
    for cand in candidates:
        data = try_parse(cand)
        if not isinstance(data, dict):
            continue
        if "related_apis" not in data:
            # 与原实现保持相近的提示
            print("解析的 JSON 缺少 'related_apis' 键")
            continue
        apis = data.get("related_apis")
        if isinstance(apis, list):
            results.extend(apis)

    if not results:
        print("没有找到任何 JSON 代码块或包含 related_apis 的对象")
        return []

    # 去重并保持顺序
    out, s = [], set()
    for a in results:
        if a not in s:
            out.append(a)
            s.add(a)
    return out


def review_pick_best(state):
    """
    从多个 API 方案中选择最优的一个
    """
    mashup = state["mashup"]
    candidates = state["candidate_apis"]
    options = state["related_apis_options"]
    prompt = state["prompt"]

    user_input = json.dumps(
        {
            "mashup": mashup,
            "candidate_apis": candidates,
            "options": options,
        },
        ensure_ascii=False,
    )

    messages = [
        {
            "role": "system",
            "content": prompt + "\n" + REVIEW_PROMPT,
        },
        {"role": "user", "content": user_input},
    ]

    if CLIENT is None or MODEL is None:
        raise RuntimeError("LLM client not initialized. Call setup_llm_client() first.")

    start_time = time.time()
    completion = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,  # 审查必须 deterministic
        stream=False
    )
    duration = time.time() - start_time
    prompt_tokens, completion_tokens, total_tokens = _extract_usage(completion)

    response = completion.choices[0].message.content
    print("[🔍 review_pick_best response]:", response)

    # 解析 JSON
    best_idx = 0
    confidence = 0.0
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            best_idx = int(result.get("best_index", 0))
            confidence = float(result.get("confidence", 0.0))
        else:
            print("[⚠️ 未找到 JSON，使用默认 index=0]")
    except Exception as e:
        print(f"[❌ 解析失败: {e}，使用默认 index=0]")

    # 确保 best_idx 在有效范围内
    best_idx = max(0, min(best_idx, len(options) - 1))

    call_logs = list(state.get("call_logs", []))
    call_logs.append(
        {
            "stage": "review_pick_best",
            "duration_sec": duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "best_index": best_idx,
            "confidence": confidence,
        }
    )

    return {
        **state,
        "related_apis": options[best_idx],
        "selected_index": best_idx,
        "call_logs": call_logs,
        "total_time": state.get("total_time", 0.0) + duration,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
    }


def final_sanity_checks(state):
    """
    本地规则兜底：
    1. 去掉不在 candidate 里的 API（去幻觉）
    2. 去重
    3. 限制在 top_k 范围内
    """
    candidates = state["candidate_apis"]
    recommended = state["related_apis"]
    top_k = state.get("top_k", DEFAULT_TOP_K)

    # 提取候选 API 的 title 集合
    candidate_titles = set()
    for api in candidates:
        if isinstance(api, dict) and 'title' in api:
            candidate_titles.add(api['title'])

    # 清洗推荐列表
    cleaned = []
    seen = set()

    for api in recommended:
        # 处理不同格式的 API
        title = ""
        if isinstance(api, dict) and 'title' in api:
            title = api['title']
        elif isinstance(api, str):
            title = api

        # 只保留在候选集中的、未见过的 API
        if title and title in candidate_titles and title not in seen:
            cleaned.append(title)
            seen.add(title)

    # 限制在 top_k 范围内
    cleaned = cleaned[:top_k]

    print(f"[✅ final_sanity_checks] 清洗后: {len(cleaned)} 个 API")

    return {
        **state,
        "related_apis": cleaned,
    }


# 定义状态结构
class WorkflowState(TypedDict):
    mashup: str
    candidate_apis: List[Dict]
    related_apis: List[str]
    related_apis_options: List[List[str]]  # 多个候选方案
    selected_index: int  # 选中的方案索引
    prompt: str
    top_k: int
    total_time: float
    total_tokens: int
    call_logs: List[Dict]
    iterations: int


def run_multiagent_flow(
    mashup_description, candidate_apis, prompt, top_k: int = DEFAULT_TOP_K
):
    """
    新的多智能体流程：
    select_top_apis → review_pick_best → final_sanity_checks → END
    """
    workflow = StateGraph(state_schema=WorkflowState)

    # 添加三个节点
    workflow.add_node("select_top_apis", select_top_apis)
    workflow.add_node("review_pick_best", review_pick_best)
    workflow.add_node("final_sanity_checks", final_sanity_checks)

    # 设置入口点
    workflow.set_entry_point("select_top_apis")

    # 线性流程，无分支
    workflow.add_edge("select_top_apis", "review_pick_best")
    workflow.add_edge("review_pick_best", "final_sanity_checks")
    workflow.add_edge("final_sanity_checks", END)

    graph = workflow.compile()

    prompt_with_top_k = prompt.replace("{{k}}", str(top_k))

    input_data = {
        "mashup": mashup_description,
        "candidate_apis": candidate_apis,
        "prompt": prompt_with_top_k,
        "related_apis": [],
        "related_apis_options": [],
        "selected_index": 0,
        "top_k": top_k,
        "total_time": 0.0,
        "total_tokens": 0,
        "call_logs": [],
        "iterations": 0,
    }

    result = graph.invoke(input_data)
    return result
