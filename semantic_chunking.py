# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

# print("Model loaded successfully!")


# sentence = "Our company provides health insurance to full-time employees."

# embedding = model.encode(sentence)

# print("Embedding created successfully!")
# print("Embedding shape:", embedding.shape)


# [0.025,
# 0.011
# 0.56
# ...


# 384]



from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open("sample_document.txt", "r", encoding="utf-8") as file:
    document = file.read()

print(document)

sentences = [
    sentence.strip()
    for sentence in document.replace("\n", " ").split(".")
    if sentence.strip()
]

# print(f"\nSentences:{sentences}")
for i, sentence in enumerate(sentences):
    print(f"{i}: {sentence}")

embeddings = model.encode(sentences)

print("\nEmbedding shape:")
print(embeddings.shape)


similarities = model.similarity_pairwise(
    embeddings[:-1],
    embeddings[1:]
)



print("\nAdjacent sentence similarities:")

for i, score in enumerate(similarities):
    print(
        f"{i} → {i + 1}: {score.item():.4f}"
    )

threshold = 0.5

breakpoints = []

for i, score in enumerate(similarities):
    if score < threshold:
        breakpoints.append(i + 1)

chunks = []

start = 0

for breakpoint in breakpoints:
    chunks.append(
        " ".join(sentences[start:breakpoint])
    )
    start = breakpoint

chunks.append(
    " ".join(sentences[start:])
)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)



with open("semantic_chunks.txt", "w", encoding="utf-8") as file:
    for chunk in chunks:
        file.write(chunk + "\n---CHUNK---\n")