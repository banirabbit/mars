# 🧠 MARS: Multi-Agent Collaborative Reasoning Framework for Service Recommendation

[🇨🇳 中文说明 / Chinese Version](./README_CN.md)

### 📘 Abstract

Service recommendation for mashup development faces critical challenges due to the sparse usage history of newly introduced mashups and APIs, as well as the difficulty of inferring genuine API dependencies from mashup compositions, where APIs are often co-used in a noisy and implicit manner. Traditional collaborative filtering, content-based methods, and standalone LLM-based approaches have limitations in jointly addressing these challenges in a unified manner. We propose MARS, a multi-agent collaborative recommendation framework that systematically integrates semantic alignment, structure-aware retrieval, and validation-based recommendation under a constrained candidate space.

MARS incorporates multiple algorithmic components to improve different stages of the service recommendation process. Specifically, agent-driven semantic enrichment substantially mitigates cross-representation semantic mismatch between mashups and APIs, reducing the average Jensen–Shannon distance from 0.7333 to 0.6333, while baseline methods exhibit negligible changes. Structure-aware fine-tuning captures API compositional patterns beyond surface-level semantics, and data-driven weight optimization learns the fusion weights in the hybrid retrieval stage, replacing static retrieval parameters with empirically calibrated strategies. Finally, multi-agent collaborative reasoning enhances robustness by combining diverse proposal generation with validation-based selection.

Experiments on **ProgrammableWeb** dataset demonstrate that MARS consistently outperforms representative baselines, achieving **63.31% Recall@5** compared to **58.28%** for Native RAG and **43.35%** for the best traditional method (ServeNet). The results indicate that MARS provides an effective and extensible framework for improving mashup-oriented service recommendation. Our code is available at https://github.com/banirabbit/mars.

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
  base_url: ""
  api_key: ""
  model_name: "Qwen2.5-14B-Instruct"
  max_retry_count: 5
```

---

### 📊 Output and Logging

* Logs: `logs/`
* Results: `output/`
* Vector Databases: `data/vector_db/`



