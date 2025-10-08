import json
from typing import List, Dict

import sys

from sentence_transformers import SentenceTransformer


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from tqdm import tqdm
import jieba
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from collections import Counter
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

    vectordb_dir = "./FAISS_DB"
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
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
    vector_retriever = db.as_retriever(search_kwargs={"k": 45})
    print("Embedding complete, query start:")

    bm25_retriever = BM25Retriever.from_documents(
        docs,
        k=45,
        bm25_params={"k1": 1.5, "b": 0.75},
        preprocess_func=jieba.lcut,
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
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
            if rd.metadata["title"] != question["title"]:
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

def compute_score(rank1, rank2, total, w1=1.0, w2=1.0):
    score = w1 * (total - rank1 + 1) + w2 * (total - rank2 + 1)
    return score

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
    mashup_path = "data/origin/active_mashups_data.txt"
    mashup_list = []
    api_list = []
    with open(mashup_path, "r", encoding="utf-8") as file:
        content = file.read()
        # 6424
        mashup_list = json.loads(content)
    api_path = "data/origin/active_apis_data.txt"
    with open(api_path, "r", encoding="utf-8") as file:
        content = file.read()
        api_list = json.loads(content)

    with open("data/rewrite/seed42/train_rewrite_data1202.json", "r", encoding="utf-8") as file:
        train_set = json.load(file)
    with open("data/rewrite/seed42/train_rewrite_data1202.json", "r", encoding="utf-8") as file:
        questions_origin = json.load(file)
    # questions_origin = questions_origin[:1]
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
    dataset = train_set
    questions = [m["description"] for m in questions_origin]

    answer_apis, total = rag_baseline(
        dataset, questions, question_origin=questions_origin
    )

    answer_rate = []
    result = []
    rag_truth_num = 0
    llm_truth_num = 0
    all_num = 0
    predicted_total = 0
    true_positive = 0

    hallu_answer = 0
    pbar = tqdm(total=len(answer_apis), desc="Processing LLM", colour="blue")

    start_index = 0
    hallu_answer = 0
    
    train_dataset = []

    for index, api_set in enumerate(answer_apis[start_index:], start=start_index):
        pbar.update(1)
    
        predict_apis = [api["title"] for api in api_set if api is not None]
        predict_apis = predict_apis[:50]
        save_apis = [api for api in api_set if api is not None]
        save_apis = save_apis[:50]
        
        count = 0
        max_correct_len = 0
        
        
 
        if "related_apis" in questions_origin[index]:
            truth_api = {
                question["title"]
                for question in questions_origin[index]["related_apis"]
                if question is not None
            }
            # llm处理后相同api的个数
            common_llm_apis = set(predict_apis).intersection(truth_api)

            if max_correct_len > 0:
                llm_truth_num += max_correct_len
            else:
                llm_truth_num += len(common_llm_apis)
            all_num += len(truth_api)
            print(list(truth_api))
            d = questions_origin[index]["description"]
            c = questions_origin[index]["categories"]
            # 构造一条数据
            item = {
                "mashup": f"description:{d}, categories:{c}",
                "related_apis": save_apis,
                "gold_apis": list(common_llm_apis)
            }
            train_dataset.append(item)
        
    with open("data/prompt_train/seed_42/llm_generated_dataset.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)


 
        

