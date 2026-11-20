import pandas as pd
import ollama
import numpy as np

from sentence_transformers import SentenceTransformer


# Load documents

with open("semantic_chunks.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

# print(f"Loaded {len(documents)} documents")


# Create embedding model

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Create document embeddings

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)


# Retrieval function

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


# Answer generation

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
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()
#this was existing rag

#Load the evaluation dataset
evaluation_dataset = pd.read_csv(
    "eval_dataset.csv"
)

print("\nEvaluation Dataset:")
print(evaluation_dataset)

#Run the RAG against every question
evaluation_results = []

for _, row in evaluation_dataset.iterrows():

    question = row["question"]
    reference = row["reference"]

    results = retrieve(question)

    answer = generate_answer(
        question,
        results
    )

    evaluation_results.append({
        "question": question,
        "reference": reference,
        "answer": answer
    })

    print("\nQuestion:")
    print(question)

    print("\nExpected Answer:")
    print(reference)

    print("\nRAG Answer:")
    print(answer)

    print("-" * 60)



#Simple automatic evaluation
def evaluate_answer(answer, reference):

    answer_words = set(
        answer.lower().split()
    )

    reference_words = set(
        reference.lower().split()
    )

    overlap = answer_words.intersection(
        reference_words
    )

    score = len(overlap) / len(reference_words)

    return score


print("\nEvaluation Scores:")

for record in evaluation_results:

    score = evaluate_answer(
        record["answer"],
        record["reference"]
    )

    print("\nQuestion:")
    print(record["question"])

    print(f"Score: {score:.2f}")




#Now lets move to security
secure_documents = [

    {
        "content": "Employees can claim approved business travel expenses.",
        "access": ["employee", "manager"]
    },

    {
        "content": "Travel expenses must be submitted within 30 days of completing the trip.",
        "access": ["employee", "manager"]
    },

    {
        "content": "International travel requires approval from the regional manager before booking.",
        "access": ["employee", "manager"]
    },

    {
        "content": "Employees receive 20 days of paid annual leave each year.",
        "access": ["employee", "manager"]
    },

    {
        "content": "Leave requests should normally be submitted at least five working days in advance.",
        "access": ["employee", "manager"]
    },

    {
        "content": "Managers approve leave requests based on team availability.",
        "access": ["manager"]
    }
]

#Secure retrieval
def secure_retrieve(
    query,
    user_role,
    top_k=3
):

    authorized_documents = []

    for document in secure_documents:

        if user_role in document["access"]:

            authorized_documents.append(
                document["content"]
            )

    if not authorized_documents:
        return []

    authorized_embeddings = embedding_model.encode(
        authorized_documents,
        normalize_embeddings=True
    )

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    scores = (
        authorized_embeddings @ query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "document": authorized_documents[index],
            "score": float(scores[index])
        })

    return results

#Secure answer generation
def secure_generate_answer(
    query,
    user_role
):

    results = secure_retrieve(
        query,
        user_role
    )

    if not results:

        return "No authorized information is available."

    context = "\n\n".join(
        result["document"]
        for result in results
    )

    prompt = f"""
Answer the user's question using only the provided context.

User Question:
{query}

Authorized Context:
{context}

If the information is not available in the authorized context,
say that the information is not available.

Give a concise and accurate answer.
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

    return response["message"]["content"].strip()


#Test security
query = "Who approves leave requests?"

answer = secure_generate_answer(
    query,
    "employee"
)

print("\nEmployee:")
print(answer)



answer = secure_generate_answer(
    query,
    "manager"
)

print("\nManager:")
print(answer)