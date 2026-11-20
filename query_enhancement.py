import ollama

query = "How long do employees have to submit travel expenses?"


# Multi query generation
prompt = f"""
Generate 3 alternative search queries for the following user question.



User question:
{query}

Rules:
- Preserve the original information need.
- Each query should express the same intent differently.
- Return only the 3 queries, one per line.
"""

response = ollama.chat(
    model="mistral:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

generated_queries = response["message"]["content"]

print("\nOriginal Query:")
print(query)

print("\nGenerated Queries:")
print(generated_queries)


from sentence_transformers import SentenceTransformer

with open("semantic_chunks.txt", "r", encoding="utf-8") as file:
    content = file.read()

documents = [
    chunk.strip()
    for chunk in content.split("---CHUNK---")
    if chunk.strip()
]

print("\nNumber of chunks:", len(documents))

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)

import re

queries = [
    re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
    for line in generated_queries.splitlines()
    if line.strip()
]

print("\nQueries to search:")

for i, q in enumerate(queries, start=1):
    print(f"{i}. {q}")



multi_query_results = []

for q in queries:

    query_embedding = embedding_model.encode(
        q,
        normalize_embeddings=True
    )

    scores = document_embeddings @ query_embedding

    ranking = sorted(
        range(len(documents)),
        key=lambda i: scores[i],
        reverse=True
    )

    multi_query_results.append(
        (ranking, scores)
    )


for query_number, (q, result) in enumerate(
    zip(queries, multi_query_results),
    start=1
):

    ranking, scores = result

    print(f"\n--- Results for Query {query_number} ---")
    print(q)

    for rank, index in enumerate(
        ranking[:3],
        start=1
    ):
        print(
            f"{rank}. Score: {scores[index]:.4f}"
        )
        print(documents[index])


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, doc_index in enumerate(ranking, start=1):
            scores[doc_index] = scores.get(doc_index, 0) + (
                1 / (k + rank)
            )

    return sorted(
        scores,
        key=scores.get,
        reverse=True
    ), scores

rankings_only = [
    ranking
    for ranking, scores in multi_query_results
]
multi_query_rrf_ranking, multi_query_rrf_scores = (
    reciprocal_rank_fusion(
        rankings_only
    )
)

print("\nMulti-Query Combined Results:")

for rank, index in enumerate(
    multi_query_rrf_ranking[:5],
    start=1
):
    print(
        f"{rank}. RRF Score: "
        f"{multi_query_rrf_scores[index]:.6f}"
    )
    print(documents[index])



# #HyDe


hyde_prompt = f"""
Write a short hypothetical answer to this question.

Question:
{query}

Use only 2-3 sentences.
Return only the answer.
"""

response = ollama.chat(
    model="mistral:latest",
    messages=[
        {
            "role": "user",
            "content": hyde_prompt
        }
    ]
)

hypothetical_document = response["message"]["content"]

print("\nHypothetical Document:")
print(hypothetical_document)

# #Embed the Hypothetical Document
hyde_embedding = embedding_model.encode(
    hypothetical_document,
    normalize_embeddings=True
)
# #Search the real semantic chunks

hyde_scores = document_embeddings @ hyde_embedding
hyde_ranking = sorted(
    range(len(documents)),
    key=lambda i: hyde_scores[i],
    reverse=True
)
print("\nHyDE Results:")

for rank, index in enumerate(
    hyde_ranking[:5],
    start=1
):
    print(
        f"{rank}. Score: {hyde_scores[index]:.4f}"
    )
    print(documents[index])