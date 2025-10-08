# src/multiagent_recommender.py

import ast
import json
import re
from typing import Dict, List, Tuple, TypedDict
from langgraph.graph import StateGraph, END
import os
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

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


def select_top_apis(state):
    apis = state["candidate_apis"]
    mashups = state["mashup"]
    core_prompt = state["prompt"]
    feedback = state.get("feedback_reason", "")

    prompt_payload = {"mashup": mashups, "candidate_apis": apis}

    if feedback:
        prompt_payload["feedback"] = feedback

    user_prompt = json.dumps(prompt_payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": core_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if CLIENT is None or MODEL is None:
        raise RuntimeError("LLM client not initialized. Call setup_llm_client() first.")

    completion = CLIENT.chat.completions.create(
        model=MODEL, messages=messages, stream=False
    )

    response = completion.choices[0].message.content
    print(response)
    return {
        **state,
        "related_apis": process_json(response),
        "feedback_reason": "",  # 清除旧的反馈
    }


# def process_json(text):
#     pattern = r"```json(.*?)```"
#     matches = re.findall(pattern, text, re.DOTALL)
#     for match in matches:
#         try:
#             json_data = json.loads(match.strip())  # 转换为 JSON 对象
#             # 检查是否存在 "related_apis" 键
#             if "related_apis" not in json_data:
#                 print("解析的 JSON 缺少 'related_apis' 键")
#                 return []
#             return json_data["related_apis"]
#         except json.JSONDecodeError as e:
#             print("解析失败:", e)
#             return []
#         except Exception as e:
#             print(f"没有找到JSON的开始或结束符: {e}")
#             return []
#     print("没有找到任何 JSON 代码块")
#     return []

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


def check_recommendation_validity(state) -> Tuple[Dict, str]:
    mashup = state["mashup"]
    candidates = state["candidate_apis"]
    recommended = state["related_apis"]
    prompt = state["prompt"]
    retry_count = state.get("retry_count", 0)

    user_input = json.dumps(
        {
            "mashup": mashup,
            "candidate_apis": candidates,
            "recommended_apis": recommended,
        },
        ensure_ascii=False,
    )

    messages = [
        {
            "role": "system",
            "content": (
                prompt
                + "\nYou are an expert assistant evaluating the quality of API recommendations."
                "\nPlease judge whether the recommended APIs satisfy the mashup's core requirements."
                "\nIf the recommendations are valid, respond in the following JSON format:\n"
                '```json\n{"valid": true}\n```\n'
                "If the recommendations are invalid, respond as:\n"
                '```json\n{"valid": false, "reason": "<brief explanation why the APIs do not meet the requirements>"}\n```\n'
            ),
        },
        {"role": "user", "content": user_input},
    ]

    if CLIENT is None or MODEL is None:
        raise RuntimeError("LLM client not initialized. Call setup_llm_client() first.")

    completion = CLIENT.chat.completions.create(
        model=MODEL, messages=messages, stream=False
    )

    response = completion.choices[0].message.content.strip()
    print("[🔍 validity check response]:", response)

    # 提取模型输出的 JSON
    pattern = r"```json(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if result.get("valid") is True:
                return {**state, "is_retry": False}
            else:
                reason = result.get(
                    "reason",
                    "The recommended APIs do not match the mashup requirements.",
                )
                return {
                    **state,
                    "is_retry": True,
                    "retry_count": retry_count + 1,
                    "feedback_reason": reason,
                }
        except Exception as e:
            print("[❌ JSON解析失败]:", e)
            return {
                **state,
                "is_retry": True,
                "retry_count": retry_count + 1,
                "feedback_reason": "Fail to parse JSON.",
            }

    # fallback，如果没有正确格式，就默认结束
    return {**state, "is_retry": False}

def revise_prompt(state):
    mashup = state["mashup"]
    original_prompt = state["prompt"]
    feedback = state.get("feedback_reason", "")
    previous_apis = state.get("related_apis", [])

    messages = [
        {"role": "system", "content": "You are an expert in writing prompts for API selection."},
        {"role": "user", "content": (
            f"The original prompt was:\n{original_prompt}\n\n"
            f"The model failed to generate useful APIs. Feedback:\n{feedback}\n\n"
            f"The mashup description and categories are {mashup}"
            f"The previous APIs were:\n{previous_apis}\n\n"
            f"Please revise the original prompt to be more helpful, specific, or constrained."
            f"Please preserve the structure of the original prompt while improving the clarity or completeness to better match the mashup's requirements."
            f"Output only the revised prompt."
        )}
    ]

    if CLIENT is None or MODEL is None:
        raise RuntimeError("LLM client not initialized. Call setup_llm_client() first.")

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    new_prompt = response.choices[0].message.content.strip()
    print("[🔧 prompt updated]:", new_prompt)

    return {
        **state,
        "prompt": new_prompt
    }
# step 3: 分支逻辑函数（决定跳转方向）
def route_based_on_validity(state) -> str:
    global MAX_RETRY_COUNT
    if state.get("retry_count", 0) >= MAX_RETRY_COUNT:
        print(f"[⚠️ 超过最大重试次数 {MAX_RETRY_COUNT}，终止流程]")
        return "end"
    return "retry" if state.get("is_retry") else "end"


# 定义状态结构
class WorkflowState(TypedDict):
    mashup: str
    candidate_apis: List[Dict]
    related_apis: List[str]
    prompt: str
    is_retry: bool
    retry_count: int


def run_multiagent_flow(mashup_description, candidate_apis, prompt):
    workflow = StateGraph(state_schema=WorkflowState)

    workflow.add_node("select_top_apis", select_top_apis)
    # workflow.add_node("revise_prompt", revise_prompt)
    workflow.add_node(
        "check_recommendation_validity", RunnableLambda(check_recommendation_validity)
    )

    workflow.set_entry_point("select_top_apis")

    # 多路分支根据校验结果跳转
    workflow.add_edge("select_top_apis", "check_recommendation_validity")
    workflow.add_conditional_edges("check_recommendation_validity", route_based_on_validity, {
    "retry": "select_top_apis",
    "end": END,
    })
    # workflow.add_conditional_edges("check_recommendation_validity", route_based_on_validity, {
    # "retry": "revise_prompt",
    # "end": END,
    # })
    # workflow.add_edge("revise_prompt", "select_top_apis")

    graph = workflow.compile()

    input_data = {
        "mashup": mashup_description,
        "candidate_apis": candidate_apis,
        "prompt": prompt,
        "related_apis": [],
        "is_retry": False,
        "retry_count": 0,
    }

    result = graph.invoke(input_data)
    return result
