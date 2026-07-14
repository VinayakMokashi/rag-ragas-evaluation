"""
RAG pipeline evaluated with RAGAS.

A minimal Retrieval-Augmented Generation pipeline that:
  1. Embeds a set of documents with HuggingFace sentence-transformer embeddings.
  2. Retrieves the most relevant document for a question via cosine similarity.
  3. Generates an answer with a Groq-hosted LLaMA model, grounded only in the
     retrieved document.
  4. Evaluates the answers with two RAGAS metrics:
       - Context Recall -- was the correct document retrieved?
       - Faithfulness   -- is the answer fully supported by the retrieved
         document (i.e. no hallucinations)?

How to run
----------
    python rag_evaluation.py

Requires a Groq API key in a ``.env`` file (``GROQ_API_KEY=...``). The first run
downloads the embedding model (all-mpnet-base-v2, ~440 MB) and caches it.
"""

import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness

# Load GROQ_API_KEY (and any other variables) from a local .env file.
load_dotenv()


class RAG:
    """A tiny retrieve-then-generate pipeline over an in-memory list of documents."""

    def __init__(self, docs, model="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model=model)
        self.embedding_model = HuggingFaceEmbeddings()
        self.docs = docs
        # Pre-embed every document once so retrieval is just a similarity lookup.
        self.doc_embeddings = self.embedding_model.embed_documents(docs)

    def get_most_relevant_docs(self, query):
        """Return the single most relevant document for ``query`` (as a 1-item list)."""
        query_embedding = self.embedding_model.embed_documents([query])[0]
        similarities = [
            self.cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def cosine_similarity(self, vec1, vec2):
        """Standard cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def generate_answer(self, query, relevant_doc):
        """Ask the LLM to answer ``query`` using only ``relevant_doc`` as context."""
        prompt = f"question: {query}\n\n Documents: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on the given documents only"),
            ("human", prompt)
        ]
        ai_message = self.llm.invoke(messages)
        return ai_message.content

    def get_eval_dataset(self, query_reference_pairs):
        """Build a RAGAS EvaluationDataset by running retrieval + generation for
        each (query, reference-answer) pair."""
        dataset = []
        for query, reference in query_reference_pairs:
            # Use self (not a module-level instance) so this works for any RAG object.
            relevant_docs = self.get_most_relevant_docs(query)
            response = self.generate_answer(query, relevant_docs)
            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": relevant_docs,
                    "response": response,
                    "reference": reference
                }
            )
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        return evaluation_dataset

    def get_eval_metrics(self, evaluation_dataset):
        """Score the dataset with RAGAS (context recall + faithfulness).

        A separate, smaller Groq model acts as the "judge" LLM that RAGAS uses to
        grade the answers.
        """
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        evaluator_llm = LangchainLLMWrapper(llm)
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
            ],
            llm=evaluator_llm
        )
        return result


if __name__ == "__main__":
    # Small knowledge base. The last entry is an unrelated distractor -- retrieval
    # should never pick it for the science questions below.
    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Vamsi works on Java"
    ]

    # (question, reference/ground-truth answer) pairs used to evaluate the pipeline.
    query_reference_pairs = [
        ("Who introduced the theory of relativity?",
         "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity."),

        ("Who was the first computer programmer?",
         "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."),

        ("What did Isaac Newton contribute to science?",
         "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics."),

        ("Who won two Nobel Prizes for research on radioactivity?",
         "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes."),

        ("What is the theory of evolution by natural selection?",
         "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."),
    ]

    rag = RAG(sample_docs)
    eval_dataset = rag.get_eval_dataset(query_reference_pairs)
    eval_metrics = rag.get_eval_metrics(eval_dataset)

    # Show each question with its retrieved doc, the LLM answer and the reference.
    for query, reference in query_reference_pairs:
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        print(f"\nQuery: {query}")
        print(f"Retrieved Doc: {relevant_docs[0]}")
        print(f"LLM Response: {response}")
        print(f"Reference Answer: {reference}")

    print("\nEvaluation Dataset:")
    print(eval_dataset.to_pandas().head())

    print("\nEvaluation Metrics Result:")
    print(eval_metrics)
