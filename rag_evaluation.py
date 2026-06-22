from ragas import EvaluationDataset
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness

load_dotenv()


class RAG:
    def __init__(self, docs, model="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model=model)
        self.embedding_model = HuggingFaceEmbeddings()
        self.docs = docs
        self.doc_embeddings = self.embedding_model.embed_documents(docs)

    def get_most_relevant_docs(self, query):
        query_embedding = self.embedding_model.embed_documents([query])[0]
        similarities = [
            self.cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def generate_answer(self, query, relevant_doc):
        prompt = f"question: {query}\n\n Documents: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on the given documents only"),
            ("human", prompt)
        ]
        ai_message = self.llm.invoke(messages)
        return ai_message.content

    def get_eval_dataset(self, query_reference_pairs):
        dataset = []
        for query, reference in query_reference_pairs:
            relevant_docs = rag.get_most_relevant_docs(query)
            response = rag.generate_answer(query, relevant_docs)
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
    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Vamsi works on Java"
    ]

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
