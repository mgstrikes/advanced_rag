from rank_bm25 import BM25Okapi

with open("semantic_chunks.txt", "r", encoding="utf-8") as file:
    content = file.read()

print(content)

documents = [
    chunk.strip()
    for chunk in content.split("---CHUNK---")
    if chunk.strip()
]

print("Number of chunks:", len(documents))

for i, document in enumerate(documents):
    print(f"\n--- Chunk {i + 1} ---")
    print(document)



query = "How long do employees have to submit travel expenses?"

#let us perform keyword search with the help of BM25

import re

stopwords = {
    "a", "an", "the", "is", "are", "was", "were",
    "do", "does", "did", "how", "what", "when",
    "where", "why", "who", "to", "of", "in",
    "on", "for", "and", "or", "with", "have",
    "has", "had", "can", "be"
}

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    return [
        word
        for word in text.split()
        if word not in stopwords
    ]



tokenized_documents = [
    tokenize(document)
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)
tokenized_query = tokenize(query)
bm25_scores = bm25.get_scores(tokenized_query)


# print("\nTokenized query:")
# print(tokenized_query)

# print("\nBM25 scores:")
# for i, score in enumerate(bm25_scores):
#     print(f"Chunk {i + 1}: {score:.4f}")


bm25_ranking = sorted(
    range(len(documents)),
    key=lambda i: bm25_scores[i],
    reverse=True
)

# print("\nBM25 Results:")

# for rank, index in enumerate(bm25_ranking[:5], start=1):
#     print(
#         f"{rank}. Score: {bm25_scores[index]:.4f}"
#     )
#     print(documents[index])


#keyword search done now let's see vector search


from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)

query_embedding = embedding_model.encode(
    query,
    normalize_embeddings=True
)

vector_scores = document_embeddings @ query_embedding


vector_ranking = sorted(
    range(len(documents)),
    key=lambda i: vector_scores[i],
    reverse=True
)

print("\nVector Search Results:")

for rank, index in enumerate(vector_ranking[:5], start=1):
    print(
        f"{rank}. Score: {vector_scores[index]:.4f}"
    )
    print(documents[index])

#searching via keyword as well as vector method done now we have to perform Rank fusion

#Let's inspect the rankings side by side
print("\nBM25 Ranking:")
for rank, index in enumerate(bm25_ranking[:5], start=1):
    print(f"{rank}. Chunk {index + 1}")

print("\nVector Ranking:")
for rank, index in enumerate(vector_ranking[:5], start=1):
    print(f"{rank}. Chunk {index + 1}")



# #Implement RRF


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

rrf_ranking, rrf_scores = reciprocal_rank_fusion(
    [
        bm25_ranking,
        vector_ranking
    ]
)

print("\nHybrid Search Results:")

for rank, index in enumerate(rrf_ranking[:5], start=1):
    print(
        f"{rank}. RRF Score: {rrf_scores[index]:.6f}"
    )
    print(documents[index])




#lets get top candidates

candidates = rrf_ranking[:5]

print("\nCandidates for re-ranking:")

for rank, index in enumerate(candidates, start=1):
    print(f"{rank}. {documents[index]}")

#cross encoding

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


pairs = [
    (query, documents[index])
    for index in candidates
]

rerank_scores = reranker.predict(pairs)


print("\nCross-Encoder Scores:")

for index, score in zip(candidates, rerank_scores):
    print(f"Score: {score:.4f}")
    print(documents[index])