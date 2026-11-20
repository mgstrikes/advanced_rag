import re
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

with open("semantic_chunks.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(documents)} documents")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)

print("Document embeddings created.")


def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "document": documents[index],
            "score": float(scores[index])
        })

    return results


# query = "How long do employees have to submit travel expenses?"

# results = retrieve(query)

# print("\nInitial Retrieval:\n")

# for i, result in enumerate(results, 1):
#     print(f"{i}. Score: {result['score']:.4f}")
#     print(result["document"])
#     print()

#upto this point we have simple RAG 
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

#lets addd and retreval evaluation funtion

def evaluate_retrieval(query, results):

    best_score = results[0]["score"]

    print(f"\nBest Similarity Score: {best_score:.4f}")

    if best_score >= 0.50:
        return "CORRECT"

    elif best_score >= 0.30:
        return "AMBIGUOUS"

    else:
        return "INCORRECT"

#lets check again
# query = "How long do employees have to submit travel expenses?"

# results = retrieve(query)
# evaluation = evaluate_retrieval(query, results)


# print("\nInitial Retrieval:\n")

# for i, result in enumerate(results, 1):
#     print(f"{i}. Score: {result['score']:.4f}")
#     print(result["document"])
#     print()
# print("Retrieval Evaluation:", evaluation)



# -----------------------------------------------------------------------
#lets add a query correction function

def correct_query(query, results):

    context = "\n\n".join(
        result["document"]
        for result in results
    )

    prompt = f"""
You are helping a RAG system improve its retrieval.

Original user query:
{query}

Retrieved information:
{context}

The retrieval was not good enough.

Rewrite the original query into a better search query.

The new query should:
- preserve the original information need
- be clear and specific
- use important keywords
- avoid adding unsupported facts

Return only the rewritten query.
"""

    response = ollama.chat(
        model="mistral:latest",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()



# #--------------------------------------------------------------------------------
# #now we can add a corrective retrieval function to correct the query if retreved results is bad
def corrective_retrieval(query):

    print("\nInitial Query:")
    print(query)

    # Step 1: Initial retrieval
    results = retrieve(query)

    print("\nInitial Retrieval:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['score']:.4f} - {result['document']}")

    # Step 2: Evaluate retrieval
    evaluation = evaluate_retrieval(query, results)

    print("\nRetrieval Evaluation:", evaluation)
    # evaluation = "AMBIGUOUS" #only uncomment when testing for ambigeous query
    # Step 3: Decide what to do
    if evaluation == "CORRECT":
        print("\nRetrieval is good enough. Using retrieved context.")
        return results

    # Step 4: Correct retrieval
    print("\nRetrieval needs correction.")

    improved_query = correct_query(query, results)

    print("\nCorrected Query:")
    print(improved_query)

    # Step 5: Retrieve again
    corrected_results = retrieve(improved_query)

    print("\nCorrected Retrieval:")

    for i, result in enumerate(corrected_results, 1):
        print(f"{i}. {result['score']:.4f} - {result['document']}")

    return corrected_results



# #So far we've seen retreval as well as corrected retrieval, but we haven't generated the answer.

def generate_answer(query, results):

    context = "\n\n".join(
        result["document"]
        for result in results
    )

    prompt = f"""
Answer the user's question using only the provided context.

User Question:
{query}

Context:
{context}

If the context does not contain enough information,
say that the information is not available.

Give a concise and accurate answer.
"""

    response = ollama.chat(
        model="mistral:latest",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()


# #lets now Run the complete CRAG pipeline
# #test 1: good query
# query = "How long do employees have to submit travel expenses?"
# #Test 2 — Poorly phrased query
# query = "after how may days of trip i can get my money"
# #Test 3 — Completely unrelated query
query = "What is the company's policy for office parking?"
final_results = corrective_retrieval(query)
answer = generate_answer(query, final_results)

print("\nFinal Answer:")
print(answer)