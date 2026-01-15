# 🧠 MARS: Multi-Agent Collaborative Reasoning Framework for Service Recommendation
[🇬🇧 English Version](./README.md)
### 📘 摘要

在 Mashup 服务开发中，服务推荐面临着关键挑战，包括新引入的 mashup 和 API 使用历史稀疏，以及难以从 mashup 组合中推断真实的 API 依赖关系——API 通常以嘈杂和隐式的方式共同使用。传统的协同过滤、基于内容的方法以及独立的大语言模型方法在统一解决这些挑战方面存在局限性。我们提出 MARS，一个多智能体协同推荐框架，系统地集成了语义对齐、结构感知检索和基于验证的推荐机制，在受约束的候选空间内工作。

MARS 融合了多个算法组件来改进服务推荐流程的不同阶段。具体而言，智能体驱动的语义增强显著缓解了 mashup 和 API 之间的跨表示语义不匹配，将平均 Jensen-Shannon 距离从 0.7333 降至 0.6333，而基线方法的变化可以忽略不计。结构感知微调捕获了超越表层语义的 API 组合模式，数据驱动的权重优化学习了混合检索阶段的融合权重，用经验校准的策略替代了静态检索参数。最后，多智能体协同推理通过结合多样化方案生成和基于验证的选择来增强鲁棒性。

在 **ProgrammableWeb** 数据集上的实验表明，MARS 在各项指标上持续优于代表性基线方法，在 Recall@5 上达到 **63.31%**，相比 Native RAG 的 **58.28%** 和最佳传统方法 ServeNet 的 **43.35%**。结果表明，MARS 为改进面向 mashup 的服务推荐提供了一个有效且可扩展的框架。我们的代码已开源：https://github.com/banirabbit/mars

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
  model_name: "Qwen2.5-14B-Instruct"
  max_retry_count: 5
```

---

### 📊 输出结果说明

* **日志文件**：`logs/`
* **推荐结果**：`output/`
* **向量数据库**：`data/vector_db/`

---