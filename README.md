# RAG Evaluation with RAGAS

A minimal **Retrieval-Augmented Generation (RAG)** pipeline that answers questions
from a small set of documents and then measures the quality of those answers using
the [RAGAS](https://docs.ragas.io/) evaluation framework.

## What's inside

| File | Description |
|------|-------------|
| [`rag_evaluation.py`](rag_evaluation.py) | The full RAG pipeline evaluated with the **RAGAS** framework (context recall + faithfulness). Described below. |
| [`retrieval_metrics.py`](retrieval_metrics.py) | A tiny, dependency-free companion that computes the classic IR metrics **Recall@k** and **Precision@k** from scratch — the foundations behind RAGAS-style retrieval scoring. |

## What `rag_evaluation.py` does

1. **Embeds** a list of documents using HuggingFace sentence-transformer embeddings.
2. **Retrieves** the most relevant document for a question via cosine similarity.
3. **Generates** an answer with a Groq-hosted LLaMA model, grounded only in the
   retrieved document.
4. **Evaluates** the answers with two RAGAS metrics:
   - **Context Recall** — was the correct document retrieved?
   - **Faithfulness** — is the answer fully supported by the retrieved document
     (i.e. no hallucinations)?

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

## Setup

Install the dependencies:

```bash
pip install ragas langchain langchain-community langchain-groq langchain-huggingface sentence-transformers numpy python-dotenv
```

Create a `.env` file in the project folder with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Run

```bash
python rag_evaluation.py
```

The first run downloads the embedding model (`all-mpnet-base-v2`, ~440 MB) and
caches it locally, so later runs start quickly.

## Output

For each query the script prints the question, the retrieved document, the LLM's
answer and the reference answer, followed by the overall evaluation scores, e.g.:

```
Evaluation Metrics Result:
{'context_recall': 1.0000, 'faithfulness': 1.0000}
```

Both metrics range from 0 to 1, where 1.0 is the best possible score.

## Bonus: retrieval metrics from scratch

[`retrieval_metrics.py`](retrieval_metrics.py) is a standalone, dependency-free
script (pure Python — nothing to install) that shows how the core retrieval
metrics are actually computed:

- **Recall@k** — of all the relevant items, how many appear in the top *k*?
- **Precision@k** — of the top *k* returned, how many are relevant?

```bash
python retrieval_metrics.py
```

It prints Recall@k and Precision@k across a range of *k* for a small worked
example, making it easy to see how each value changes as *k* grows. These are
the same ideas that underpin RAGAS's retrieval-side scoring.
