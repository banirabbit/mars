import json
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import jieba
import os

### 1. 加载数据 ###
def load_mashups(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

### 2. 计算 API 重合度 ###
def compute_api_overlap(target_apis, candidate_apis):
    target_ids = set(api["api_id"] for api in target_apis)
    candidate_ids = set(api["api_id"] for api in candidate_apis)
    if not target_ids or not candidate_ids:
        return 0.0
    intersection = target_ids & candidate_ids
    union = target_ids | candidate_ids
    return len(intersection) / len(union)

### 3. 构建特征数据集 ###
def create_dataset(mashup_data, all_mashups, bm25_retriever, vector_retriever, top_k=44):
    X = []
    y = []

    mashup_by_title = {m["title"]: m for m in all_mashups}
    visited_pairs = set()
    max_rank = top_k - 1

    for mashup in tqdm(mashup_data, desc="Generating samples"):
        query = mashup["description"]
        target_apis = mashup["related_apis"]
        target_title = mashup["title"]

        bm25_docs = bm25_retriever.get_relevant_documents(query)
        vector_docs = vector_retriever.get_relevant_documents(query)

        bm25_titles = [doc.metadata["title"] for doc in bm25_docs]
        vector_titles = [doc.metadata["title"] for doc in vector_docs]

        all_candidates = set(bm25_titles) | set(vector_titles)

        for candidate_title in all_candidates:
            if candidate_title == target_title:
                continue

            pair_key = tuple(sorted([target_title, candidate_title]))
            if pair_key in visited_pairs:
                continue
            visited_pairs.add(pair_key)

            candidate = mashup_by_title.get(candidate_title)
            if not candidate:
                continue

            # 使用排名位置转换为得分（越靠前得分越高）
            bm25_rank = bm25_titles.index(candidate_title) if candidate_title in bm25_titles else top_k
            vector_rank = vector_titles.index(candidate_title) if candidate_title in vector_titles else top_k

            # 归一化：score = 1 - rank / top_k
            bm25_score = 1 - (bm25_rank / top_k)
            vector_score = 1 - (vector_rank / top_k)

            label = compute_api_overlap(target_apis, candidate["related_apis"])

            X.append([bm25_score, vector_score])
            y.append(label)

    return X, y

### 4. 初始化检索器 ###
def init_retrievers(mashups, embed_model="output/finetuned_bge_singlegpu_2025-07-17_10-15"):
    docs = []
    for mashup in mashups:
        d = mashup["description"]
        doc = Document(page_content=d, metadata={"title": mashup["title"]})
        docs.append(doc)

    print("Split complete, embedding start")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        show_progress=True,
        model_kwargs={"trust_remote_code": True},
    )

    db = FAISS.from_documents(documents=docs, embedding=embeddings)
    db.save_local("./FAISS_DB")
    vector_retriever = db.as_retriever(search_kwargs={"k": 100})
    print("Embedding complete, query start:")

    bm25_retriever = BM25Retriever.from_documents(
        docs,
        k=100,
        bm25_params={"k1": 1.5, "b": 0.75},
        preprocess_func=jieba.lcut,
    )

    return bm25_retriever, vector_retriever

def save_dataset(X, y, filename):
    df = pd.DataFrame(X, columns=["bm25_score", "vector_score"])
    df["label"] = y
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

### 5. 主程序 ###
if __name__ == "__main__":
    # 加载数据
    train_mashups = load_mashups("data/origin/seed42/train_data.json")
    val_mashups = load_mashups("data/origin/seed42/val_data.json")
    test_mashups = load_mashups("data/rewrite/seed42/test_rewrite_data1202.json")
    all_mashups = train_mashups + val_mashups

    # 初始化检索器
    bm25_retriever, vector_retriever = init_retrievers(all_mashups)

    # 构建数据集
    X_train, y_train = create_dataset(train_mashups, train_mashups, bm25_retriever, vector_retriever)
    X_val, y_val = create_dataset(val_mashups, all_mashups, bm25_retriever, vector_retriever)


    # 保存数据集
    save_dataset(X_train, y_train, "data/weight/training_data.csv")
    save_dataset(X_val, y_val, "data/weight/validation_data.csv")