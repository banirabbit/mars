
import os
import json
from typing import List, Dict
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import json_repair
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from tqdm import tqdm
import jieba
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from collections import Counter
from multiagent_recall import run_multiagent_flow
from utils.NormalizedDCG import NormalizedDCG
from sklearn.metrics.pairwise import cosine_similarity
import re


def rag_baseline(
    mashups: List[Dict], questions: List[str], mode="simi", question_origin=[]
):
    docs = []
    for mashup in mashups:
        d = mashup["description"]
        content = d
        doc = Document(page_content=content, metadata={"title": mashup["title"]})
        docs.append(doc)
    print("Split complete, embedding start")
    # vectordb_dir = (
    #     "./servicegraphreco/graph/FAISS_DB"
    # )
    vectordb_dir = "./servicegraphreco/graph/FAISS_DB"
    # embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    # embed_model = "/home/yinzijie/code/servicerag/src/retrieval/output/finetuned_bge_singlegpu_2025-07-07_18-47"
    # embed_model = "BAAI/bge-base-en-v1.5"
    # embed_model = "/home/yinzijie/code/servicerag/src/retrieval/output/finetuned_bge_singlegpu_2025-07-17_10-15"
    embed_model = "/home/yinzijie/code/servicerag/src/retrieval/output/finetuned_bge_singlegpu_2025-09-18_19-19_0.7"
    # embed_model = "/home/yinzijie/code/servicerag/src/retrieval/output/finetuned_bge_singlegpu_2025-09-18_19-48_0.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        show_progress=True,
        model_kwargs={
            "trust_remote_code": True,
        },
    )
    db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    db.save_local(vectordb_dir)
    docs = docs
    vector_retriever = db.as_retriever(search_kwargs={"k": 100})
    print("Embedding complete, query start:")

    bm25_retriever = BM25Retriever.from_documents(
        docs,
        k=100,
        bm25_params={"k1": 1.5, "b": 0.75},
        preprocess_func=jieba.lcut,
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5157,0.4843]
    )
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

    compressor = CrossEncoderReranker(model=rerank_model, top_n=90)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    rerank_answers = []

    for question in tqdm(question_origin):
        d = question["description"]
        content = d
        relevant_docs = compression_retriever.invoke(content)

        rerank_mashups = []
        for rd in relevant_docs:
            rerank_mashups.append(rd.metadata["title"])
        rerank_answers.append(rerank_mashups)
    # 对api进行重排

    top_n_mashup_apis = get_topn_mashup_api(rerank_answers, mashups)
    rerank_answers = []
    total_characters = 0
    pbar = tqdm(total=len(top_n_mashup_apis), desc="Processing LLM", colour="blue")
    api_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for index, api_doc_set in enumerate(top_n_mashup_apis):
        pbar.update(1)
        if len(api_doc_set) > 0:
            rerank_apis = []  # 存储推荐的API

            # 获取当前Mashup的标签
            cat_list = question_origin[index]["categories"]
            cat_list = ", ".join(cat_list)
            tag_list = question_origin[index]["tags"]
            cat_list = cat_list + ", ".join(tag_list)
            # 对Mashup标签生成嵌入
            mashup_embedding = api_embed_model.encode(cat_list)

            # 对每个API文档集进行嵌入生成
            for api_doc in api_doc_set:
                api_tags = api_doc["tags"]
                api_tags = ", ".join(api_tags)

                api_embedding = api_embed_model.encode(api_tags)

                similarity_score = cosine_similarity(
                    [mashup_embedding], [api_embedding]
                )[0][0]
                rerank_apis.append((api_doc, similarity_score))

            ordered_apis = api_doc_set[:40]
            sorted_apis = sorted(rerank_apis, key=lambda x: x[1], reverse=True)
            top_10_apis_by_similarity = [
                api[0] for api in sorted_apis if api[0] not in ordered_apis
            ]
            top_10_apis_by_similarity = top_10_apis_by_similarity[:10]
            final_api_list = ordered_apis + top_10_apis_by_similarity
            final_api_list = final_api_list[:50]
        
            rerank_answers.append(final_api_list)
        else:
            rerank_apis = []
            rerank_answers.append(rerank_apis)
    return rerank_answers, total_characters

def get_topn_mashup_api(rerank_answers, mashups, top_n=50):
    api_docs = []
    for index, mashup_set in enumerate(rerank_answers):
        # print("index::", questions[index])
        api_doc_set = []
        # mashup_set = mashup_set[:top_n]
        for mashup in mashup_set:
            # print("mashup::", mashup)
            for origin_mashup in mashups:
                if origin_mashup["title"] == mashup:
                    if (
                        "related_apis" in origin_mashup
                        and origin_mashup["related_apis"]
                    ):
                        related_apis = origin_mashup["related_apis"]

                        for api in related_apis:
                            if (
                                api is not None
                                and isinstance(api, dict)
                                and "title" in api
                                and "tags" in api
                            ):
                                api_json = {
                                    "title": api["title"],
                                    "tags": api["tags"],
                                 
                                    
                                }
                                api_doc_set.append(api_json)
        # Step 1: 统计每个 title 出现的次数
        title_counts = Counter(obj["title"] for obj in api_doc_set)
        # Step 2: 根据 title 去重，保留第一个出现的对象
        unique_objects = {}
        for obj in api_doc_set:
            if obj["title"] not in unique_objects:
                unique_objects[obj["title"]] = obj
        # Step 3: 根据重复次数从大到小排序
        sorted_objects = sorted(
            unique_objects.values(),
            key=lambda x: title_counts[x["title"]],
            reverse=True,
        )
        api_docs.append(sorted_objects)
    return api_docs

def call_with_messages_multi_agent(question, answer_apis, function=[]):
    with open("/home/yinzijie/code/servicerag/src/prompt/qwen_best.txt", "r", encoding="utf-8") as file:
        prompt_origin = file.read()

    d = question["description"]
    c = question["categories"]

    # ✨ Run multi-agent pipeline in English
    mashup_text = f"description:{d}, categories:{c}"
    agent_result = run_multiagent_flow(mashup_text, answer_apis, prompt_origin)

    return agent_result

def process_json(text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            json_data = json.loads(match.strip())  # 转换为 JSON 对象
            print(json_data)
             # 检查是否存在 "related_apis" 键
            if "related_apis" not in json_data:
                print("解析的 JSON 缺少 'related_apis' 键")
                return {"related_apis": []}
            return json_data
        except json.JSONDecodeError as e:
            print("解析失败:", e)
            return {"related_apis":[]}
        except Exception as e:
            print(f"没有找到JSON的开始或结束符: {e}")
            return {"related_apis":[]}
    print("没有找到任何 JSON 代码块")
    return {"related_apis": []}


def count_wrong_answer(api_list, apis):
    if len(api_list) == 0:
        return 10
    wrong_answer = 0
    apis = [a['title'] for a in apis]
    for api in api_list:
        if api != "":
            if api not in apis:
                wrong_answer += 1
    return wrong_answer
    
if __name__ == "__main__":
    print("mashup_cat plus 1")
    # 1.重复出现的api（频率） 2.mashup之间是否共享api，考虑边的权重，构建图，划分社区，做summary 3.让大模型思考api有哪些功能构成
    mashup_path = "/home/yinzijie/code/servicerag/data/origin/active_mashups_data.txt"
    mashup_list = []
    api_list = []
    with open(mashup_path, "r", encoding="utf-8") as file:
        content = file.read()
        # 6424
        mashup_list = json.loads(content)
    api_path = "/home/yinzijie/code/servicerag/data/origin/active_apis_data.txt"
    with open(api_path, "r", encoding="utf-8") as file:
        content = file.read()
        api_list = json.loads(content)

    with open("/home/yinzijie/code/servicerag/data/rewrite/seed42/train_rewrite_data1202.json", "r", encoding="utf-8") as file:
        train_set = json.load(file)
    with open("/home/yinzijie/code/servicerag/data/rewrite/seed42/test_rewrite_data1202.json", "r", encoding="utf-8") as file:
        questions_origin = json.load(file)

    answer_list = []
    for q in questions_origin:
        answer = {}
        apis = []
        answer["title"] = q["title"]
        if "related_apis" in q and q["related_apis"]:
            related_apis = q["related_apis"]
            for api in related_apis:
                if isinstance(api, dict) and api is not None:
                    apis.append(api["title"])
        answer["answers"] = apis
        answer_list.append(answer)
    # questions_origin = questions_origin[:15]

    dataset = train_set
    questions = [m["description"] for m in questions_origin]

    # answer_apis, total = rag_baseline(
    #     dataset, questions, question_origin=questions_origin
    # )
    # with open("./42_07.json", "w", encoding="utf-8") as file:
    #     json.dump(answer_apis, file, ensure_ascii=False, indent=4)
    with open("./42_05.json", "r", encoding="utf-8") as file:
        answer_apis = json.load(file)

    answer_rate = []
    result = []
    rag_truth_num = 0
    llm_truth_num = 0
    all_num = 0
    predicted_total = 0
    true_positive = 0

    hallu_answer = 0
    pbar = tqdm(total=len(answer_apis), desc="Processing LLM", colour="blue")
    total_ndcg = 0
    
    log_file = "/home/yinzijie/code/servicerag/log/pos05.json"
    print(log_file)

    # 加载日志
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_data = json.load(f)
        
    else:
        log_data = {
            "last_index": 0,
            "hallu_answer": 0,
            "llm_truth_num": 0,
            "all_num": 0,
            "total_ndcg": 0.0,
            "reco_apis": []
        }

    # 恢复状态
    start_index = log_data["last_index"]
    hallu_answer = log_data["hallu_answer"]
    llm_truth_num = log_data["llm_truth_num"]
    all_num = log_data["all_num"]
    total_ndcg = log_data["total_ndcg"]
    reco_apis = log_data["reco_apis"]
    for index, api_set in enumerate(answer_apis[start_index:], start=start_index):
        pbar.update(1)
    
        predict_apis = [api for api in api_set if api is not None]
        predict_apis = predict_apis[:50]
        
        count = 0
        max_correct_len = 0
        
        while True:
            llm_str = call_with_messages_multi_agent(questions_origin[index], predict_apis)
            llm_predict_apis = llm_str.get("related_apis", [])[:10]
            print(llm_predict_apis)
            unique_api_doc_set = []
            seen_titles = set()
            for obj in llm_predict_apis:
                if isinstance(obj, dict):
                    if 'title' in obj:
                        title = obj['title']
                        if title not in seen_titles:
                            seen_titles.add(title)
                            unique_api_doc_set.append(obj['title'])
                elif isinstance(obj, str):
                    if obj not in seen_titles:
                        seen_titles.add(obj)
                        unique_api_doc_set.append(obj)
            llm_predict_apis = unique_api_doc_set
            single_hallu = count_wrong_answer(llm_predict_apis, predict_apis)
            if "related_apis" in questions_origin[index]:
                truth_api = {
                    question["title"]
                    for question in questions_origin[index]["related_apis"]
                    if question is not None
                }
                # llm处理后相同api的个数
                common_llm_apis = set(llm_predict_apis).intersection(truth_api)
                if len(common_llm_apis) > max_correct_len:
                    max_correct_len = len(common_llm_apis)
            count += 1
            if count >= 15 or single_hallu < 1:
                break
        hallu_answer += single_hallu
        reco_apis.append(llm_predict_apis)
        if "related_apis" in questions_origin[index]:
            truth_api = {
                question["title"]
                for question in questions_origin[index]["related_apis"]
                if question is not None
            }
            # llm处理后相同api的个数
            common_llm_apis = set(llm_predict_apis).intersection(truth_api)

            if max_correct_len > 0:
                llm_truth_num += max_correct_len
            else:
                llm_truth_num += len(common_llm_apis)
            all_num += len(truth_api)
            ndcg_calculator = NormalizedDCG(10)
            print(list(truth_api))
            ndcg = ndcg_calculator.calculate_ndcg(llm_predict_apis, list(truth_api))
            total_ndcg += ndcg
        
        # 更新日志
        log_data["last_index"] = index + 1
        log_data["hallu_answer"] = hallu_answer
        log_data["llm_truth_num"] = llm_truth_num
        log_data["all_num"] = all_num
        log_data["total_ndcg"] = total_ndcg
        log_data["reco_apis"] = reco_apis

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)

    predicted_total = len(questions) * 10
    llm_recall = llm_truth_num / all_num if all_num > 0 else 0
    llm_precision = llm_truth_num / predicted_total if predicted_total > 0 else 0
    llm_hallu = hallu_answer / predicted_total if predicted_total > 0 else 0
    llm_ndcg = total_ndcg / len(answer_apis)
    if llm_recall + llm_precision > 0:
        llm_f1_score = 2 * (llm_recall * llm_precision) / (llm_precision + llm_recall)
    else:
        llm_f1_score = 0

    print("llm 召回率", llm_recall)
    print("llm 准确率：", llm_precision)
    print("llm f1的值", llm_f1_score)
    print("llm NDCG的值", llm_ndcg)
    print(f"llm出现幻觉的次数{hallu_answer}, 概率{llm_hallu}")
