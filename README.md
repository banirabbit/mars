# 🧠 MARS: Multi-Agent Collaborative Reasoning Framework for Service Recommendation

[🇨🇳 中文说明 / Chinese Version](./README_CN.md)

### 📘 Abstract

Service recommendation for mashup development faces critical challenges such as **cold-start problems** and the difficulty of **identifying genuine API dependencies** within noisy co-occurrence data. Traditional methods—including collaborative filtering, content-based recommendation, and simple LLM-based retrieval—fail to address these issues effectively.

We propose **MARS (Multi-Agent Collaborative Reasoning System)**, a framework that **extends Retrieval-Augmented Generation (RAG)** with systematic algorithmic innovations rather than relying on prompt engineering alone.

MARS introduces four major advancements:

1. **Agent-driven semantic enrichment** reduces the semantic gap between mashups and APIs from 0.7311 to 0.6448 (Δ=0.0863, measured by Jensen–Shannon divergence), while baselines show negligible improvement.
2. **Structure-aware fine-tuning** learns compositional API patterns via contrastive learning, capturing dependencies beyond surface semantics.
3. **Data-driven retrieval optimization** replaces static fusion weights with adaptive, learned parameters validated on real data.
4. **Multi-agent collaborative reasoning** implements a closed-loop “recommend–validate–revise” mechanism, ensuring consensus and factual consistency across agents.

Experiments on the **ProgrammableWeb** dataset show that MARS achieves **61.22% Recall@5**, outperforming **Native RAG (58.28%)** and **ServeNet (43.35%)**, corresponding to **5.0%** and **41.1%** relative improvements, respectively.

---

### 🧩 Project Overview

```
mars/
├── main.py                     # Entry point with CLI interface
├── config.yaml                 # Global configuration file
├── data/                       # Raw and processed data
├── logs/                       # Execution logs
├── output/                     # Generated results
├── prompts/                    # Prompt templates
└── src/                        # Source code
    ├── config.py               # Configuration manager
    ├── rag_service.py          # RAG-based retrieval service
    ├── api_recommendation_service.py # Multi-agent API recommendation
    ├── evaluation_service.py   # Evaluation metrics and reporting
    ├── main_orchestrator.py    # Orchestration of the full pipeline
    ├── qwen_multiagent.py      # Legacy-compatible main agent file
    ├── multiagent_recall.py    # Multi-agent recall module
    ├── utils/                  # Utility functions
    └── preprocess/             # Data preprocessing scripts
```

---

### ⚙️ Installation

Environment Setup

Please use Python 3.10 or higher.
You can install the required dependencies in one of the following ways:
```bash
# Option 1: Install directly
pip install sentence-transformers langchain-community faiss-cpu scikit-learn tqdm jieba pyyaml openai langgraph

# Option 2: Install from requirements file
pip install -r requirements.txt
```
Download the Pretrained Embedding Model

```bash
git clone https://huggingface.co/xiaotubani/mars-finetune
```
---

### 🧠 How to Run

```bash
# Run the full MARS pipeline
python main.py

# Check configuration validity
python main.py --config-check

# Demonstrate API usage and workflow
python main.py --demo

# Run only the retrieval module
python main.py --rag-only

# Evaluate existing prediction results
python main.py --eval-only results.json
```

---

### 🧾 Configuration Example (`config.yaml`)

```yaml
paths:
  mashup_data_path: "data/origin/active_mashups_data.txt"
  api_data_path: "data/origin/active_apis_data.txt"
  train_data_path: "data/rewrite/seed42/train_rewrite_data1202.json"
  test_data_path: "data/rewrite/seed42/test_rewrite_data1202.json"

llm:
  base_url: "http://192.168.1.101:30111/v1"
  api_key: "loopinnetwork"
  model_name: "Qwen2.5-14B-Instruct"
  max_retry_count: 5
```

---

### 📊 Output and Logging

* Logs: `logs/`
* Results: `output/`
* Vector Databases: `data/vector_db/`



