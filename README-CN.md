# 🧠 MARS: Multi-Agent Collaborative Reasoning Framework for Service Recommendation
[🇬🇧 English Version](./README.md)
### 📘 摘要

在 Mashup 服务开发中，服务推荐面临着**冷启动问题**以及**在噪声共现数据中识别真实 API 依赖关系**的挑战。传统的协同过滤、内容推荐及基于大模型的简单检索增强方法难以有效解决这些问题。

本文提出 **MARS（Multi-Agent Collaborative Reasoning System，多智能体协同推理系统）**，它在传统 RAG 框架基础上引入系统性的算法创新，突破了仅依赖提示工程的局限。

MARS 的核心创新包括：

1. **智能体驱动的语义增强**，显著缩小 Mashup 与 API 之间的语义鸿沟（Jensen-Shannon 散度从 0.7311 降至 0.6448）。
2. **结构感知的微调方法**，通过对比学习捕捉 API 组合模式，识别深层依赖关系。
3. **数据驱动的权重优化机制**，以学习型融合策略替代静态检索参数。
4. **多智能体协同推理机制**，引入闭环的“推荐—验证—修正”迭代流程，使不同智能体达成一致性判断。

在 **ProgrammableWeb** 数据集上的实验表明，MARS 在 Recall@5 上达到 **61.22%**，相比 Native RAG（58.28%）和 ServeNet（43.35%）分别提升 **5.0%** 和 **41.1%**，验证了其优越的检索与推理能力。

---

### 📂 项目结构

```
mars/
├── main.py                     # 主入口文件
├── config.yaml                 # 全局配置文件
├── data/                       # 数据目录
├── logs/                       # 日志目录
├── output/                     # 输出目录
├── prompts/                    # 提示模板目录
└── src/                        # 源代码目录
    ├── config.py               # 配置管理模块
    ├── rag_service.py          # RAG 检索模块
    ├── api_recommendation_service.py # 多智能体推荐模块
    ├── evaluation_service.py   # 评估模块
    ├── main_orchestrator.py    # 主协调模块
    ├── qwen_multiagent.py      # 兼容旧版主程序文件
    ├── multiagent_recall.py    # 多智能体召回模块
    ├── utils/                  # 工具函数
    └── preprocess/             # 数据预处理
```

---

### ⚙️ 环境依赖安装


请使用 Python 3.10 及以上版本。
安装所需依赖包可通过以下两种方式完成：
```bash
# 方式一：直接安装依赖

pip install sentence-transformers langchain-community faiss-cpu scikit-learn tqdm jieba pyyaml openai langgraph

# 方式二：通过 requirements.txt 安装
pip install -r requirements.txt
```
下载训练好的嵌入模型

```bash
git clone https://huggingface.co/xiaotubani/mars-finetune
```
---

### 🚀 运行方式

```bash
# 运行完整推荐流程
python main.py

# 检查配置文件
python main.py --config-check

# 查看 API 使用演示
python main.py --demo

# 仅运行检索模块
python main.py --rag-only

# 对推荐结果进行评估
python main.py --eval-only results.json
```

---

### 🧩 配置文件示例

```yaml
paths:
  mashup_data_path: "data/origin/active_mashups_data.txt"
  api_data_path: "data/origin/active_apis_data.txt"
  train_data_path: "data/rewrite/seed42/train_rewrite_data1202.json"
  test_data_path: "data/rewrite/seed42/test_rewrite_data1202.json"

llm:
  base_url: ""
  api_key: ""
  model_name: ""
  max_retry_count: 5
```

---

### 📊 输出结果说明

* **日志文件**：`logs/`
* **推荐结果**：`output/`
* **向量数据库**：`data/vector_db/`

---