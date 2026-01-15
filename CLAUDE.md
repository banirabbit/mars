# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MARS (Multi-Agent Collaborative Reasoning System) is a research framework for API service recommendation in mashup development. It combines Retrieval-Augmented Generation (RAG) with multi-agent collaborative reasoning to recommend APIs based on mashup descriptions.

## Key Commands

### Running the System

```bash
# Run the full MARS pipeline (RAG + Multi-agent recommendation + Evaluation)
python main.py

# Run only RAG retrieval
python main.py --rag-only

# Run evaluation on pre-computed results
python main.py --eval-only results.json

# Check configuration validity
python main.py --config-check

# Show API demonstration
python main.py --demo
```

### Development Commands

```bash
# Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# Download the pretrained embedding model (required for first run)
git clone https://huggingface.co/xiaotubani/mars-finetune
```

## Architecture Overview

### Core Pipeline Flow

The system follows a three-stage pipeline:

1. **RAG Retrieval** (`src/rag_service.py`): Retrieves candidate APIs using hybrid search (FAISS vector search + BM25) with cross-encoder reranking
2. **Multi-Agent Recommendation** (`src/api_recommendation_service.py` + `src/multiagent_recall.py`): Uses multi-agent LLM system to refine recommendations through collaborative reasoning
3. **Evaluation** (`src/evaluation_service.py`): Calculates metrics (Precision, Recall, F1, NDCG) and logs results

The orchestrator (`src/main_orchestrator.py`) coordinates all components and manages data flow.

### Configuration System

All system parameters are managed through `config.yaml` and loaded via `src/config.py`. The configuration uses dataclasses and supports:
- Model paths (embedding, reranking, API similarity models)
- Retrieval parameters (k values, weights, BM25 parameters)
- File paths (data, prompts, outputs)
- Evaluation parameters (retry attempts, thresholds)
- System settings (debug, caching, workers)

Paths in `config.yaml` are relative to project root and automatically resolved to absolute paths by the Config class.

### Multi-Agent System

The multi-agent recommendation system (`src/multiagent_recall.py`) implements a "recommend-validate-revise" loop using LangGraph:
- Multiple agents collaborate to reach consensus on API recommendations
- Integrates with LLM API (configured via `config.yaml` under `llm` section)
- Supports retry mechanisms for improved accuracy

### RAG Implementation Details

The RAG service uses a hybrid retrieval approach:
1. **Vector Search**: FAISS with fine-tuned BGE embeddings
2. **BM25**: Keyword-based retrieval with configurable parameters
3. **Ensemble**: Weighted fusion (weights learned via `src/preprocess/find_weight/`)
4. **Reranking**: Cross-encoder model for final ranking
5. **API Extraction**: Frequency-based API extraction from retrieved mashups

### Data Flow

- **Input**: Mashup descriptions with categories (from `data/origin/` or `data/rewrite/`)
- **Knowledge Base**: Training mashups used to build vector database
- **Candidates**: RAG retrieves ~50 candidate APIs per query
- **Output**: Top-N API recommendations (typically 10) with evaluation metrics

## Important Implementation Notes

### Offline Mode

The system sets offline environment variables in `main.py` to prevent network requests to HuggingFace Hub:
```python
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
```

Ensure models are downloaded locally before running in offline mode.

### Caching Strategy

- RAG results are cached in `output/rag_cache.json` (configurable)
- Evaluation logs are incrementally saved to `logs/evaluation.json`
- Use `use_rag_cache=True` to skip re-running RAG retrieval

### Legacy Compatibility

The codebase maintains backward compatibility through wrapper functions in `src/main_orchestrator.py`:
- `rag_baseline()`: Legacy RAG interface
- `get_topn_mashup_api()`: Legacy API extraction
- `call_with_messages_multi_agent()`: Legacy multi-agent interface

Prefer using the new orchestrator-based API for new code.

### Preprocessing Pipeline

The `src/preprocess/` directory contains data preparation scripts:
- `embedder/`: Fine-tuning scripts for BGE embedding model
- `rewrite/`: Query rewriting for semantic enrichment
- `generate_prompt/`: Prompt template generation
- `find_weight/`: Learning optimal ensemble weights (BM25 vs vector)

## Configuration Tips

### LLM Configuration

Edit `config.yaml` under the `llm` section:
```yaml
llm:
  base_url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  model_name: "Qwen2.5-14B-Instruct"  # or other OpenAI-compatible models
  max_retry_count: 5
```

### Retrieval Tuning

Key parameters in `config.yaml`:
- `initial_k`: Initial retrieval count (default: 100)
- `rerank_top_n`: Number to rerank (default: 90)
- `final_api_limit`: Final candidates for LLM (default: 50)
- `bm25_weight` / `vector_weight`: Ensemble fusion weights

### Model Paths

Update model paths in `config.yaml` to match your local setup:
```yaml
model:
  embed_model_path: "src/embedder/finetuned_bge_singlegpu_2025-07-07_18-47"
  rerank_model_name: "BAAI/bge-reranker-v2-m3"
  api_embed_model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

## Testing Subset

Note: The orchestrator limits test data to first 5 samples by default (see `src/main_orchestrator.py:89`):
```python
test_data = test_data[:5]
```

Remove or modify this line for full dataset evaluation.

Service recommendation for mashup development faces critical challenges due to the sparse usage history of newly introduced mashups and APIs, as well as the difficulty of inferring genuine API dependencies from mashup compositions, where APIs are often co-used in a noisy and implicit manner. Traditional collaborative filtering, content-based methods, and standalone LLM-based approaches have limitations in jointly addressing these challenges in a unified manner. We propose MARS, a multi-agent collaborative recommendation framework that systematically integrates semantic alignment, structure-aware retrieval, and validation-based recommendation under a constrained candidate space. 

MARS incorporates multiple algorithmic components to improve different stages of the service recommendation process. Specifically, agent-driven semantic enrichment substantially mitigates cross-representation semantic mismatch between mashups and APIs, reducing the average Jensen–Shannon distance from 0.7333 to 0.6333, while baseline methods exhibit negligible changes %(average absolute change of 0.0009)
. Structure-aware fine-tuning captures API compositional patterns beyond surface-level semantics, and data-driven weight optimization learns the fusion weights in the hybrid retrieval stage, replacing static retrieval parameters with empirically calibrated strategies. Finally, multi-agent collaborative reasoning enhances robustness by combining diverse proposal generation with validation-based selection. Experiments on 

ProgrammableWeb dataset demonstrate that MARS consistently outperforms representative baselines, achieving 63.31\% Recall@5 compared to 58.28\% for Native RAG and 43.35\% for the best traditional method (ServeNet). The results indicate that MARS provides an effective and extensible framework for improving mashup-oriented service recommendation. Our code is available at https://github.com/banirabbit/mars.