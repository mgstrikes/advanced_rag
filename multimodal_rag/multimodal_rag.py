import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
import ollama

with open("travel_policy.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

# print("Text loaded.")
# print(text_content)

#Load the table:
table = pd.read_csv("travel_policy.csv")

# print("\nTable:")
# print(table)

#load the image:
image = Image.open("travel_policy.png")

# print("\nImage loaded.")
# print("Image size:", image.size)
#-----------------------------------------------------------------------------------

#Create the text representation
text_model = SentenceTransformer("all-MiniLM-L6-v2")
text_embedding = text_model.encode(
    text_content,
    normalize_embeddings=True
)

# print("\nText embedding shape:", text_embedding.shape)

#-----------------------------------------------------------------------------------

#Represent the table
str(table)

table_records = table.to_dict(orient="records")

# print("\nStructured Table:")
# for row in table_records:
#     print(row)
# #-----------------------------------------------------------------------------------
# #Creating a searchable table representation
table_text = []

for row in table_records:
    row_text = (
        f"Travel Type: {row['Travel Type']}; "
        f"Limit: {row['Limit']}; "
        f"Approval: {row['Approval']}"
    )

    table_text.append(row_text)

# print("\nSearchable Table Representation:")

for row in table_text:
    print(row)



# #Embed the table representation
table_embeddings = text_model.encode(
    table_text,
    normalize_embeddings=True
)

# print("\nTable embedding shape:", table_embeddings.shape)

# #let's now Load a multimodal model for the image
clip_model = SentenceTransformer(
    "clip-ViT-B-32"
)
image_embedding = clip_model.encode(
    image,
    normalize_embeddings=True
)

# print("\nImage embedding shape:", image_embedding.shape)
# #-----------------------------------------------------------------------------------
# #let's Create our multimodal knowledge base
knowledge_base = {
    "text": {
        "content": text_content,
        "embedding": text_embedding
    },
    "table": {
        "content": table_records,
        "searchable_text": table_text,
        "embedding": table_embeddings
    },
    "image": {
        "content": "travel_policy.png",
        "embedding": image_embedding
    }
}

# print("\nMultimodal Knowledge Base Created.")

# print("\nAvailable Sources:")

# for modality, data in knowledge_base.items():
#     print(f"- {modality.capitalize()}")

#------------------------------------------------------------------------------
#Test Text Retrieval
#------------------------------------------------------------------------------
# # Step 13 — Test text retrieval

# query = "How long do employees have to submit travel expenses?"

# # # Create embedding for the query
# query_embedding = text_model.encode(query)

# # # Calculate similarity with the text source
# similarity = query_embedding @ knowledge_base["text"]["embedding"]

# print("\nQuery:")
# print(query)

# print("\nText Retrieval Result:")
# print(f"Similarity Score: {similarity:.4f}")
# print("Source: Text")
# print("Content:")
# print(knowledge_base["text"]["content"])
# #------------------------------------------------------------------------------
# #check final answer
# #------------------------------------------------------------------------------
# context = knowledge_base["text"]["content"]
# prompt = f"""
# Answer the user's question using only the provided context.

# Question:
# {query}

# Context:
# {context}

# Give a concise answer.
# """

# response = ollama.chat(
#     model="mistral:latest",
#     messages=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
# )

# print("\nFinal Answer:")
# print(response["message"]["content"])

#------------------------------------------------------------------------------
#Test Table Retrieval
#------------------------------------------------------------------------------
# Step 14 — Test table retrieval

query = "What is the expense limit for international travel?"

# # Create embedding for the query
query_embedding = text_model.encode(query)

# # Calculate similarity with each table record
similarities = query_embedding @ knowledge_base["table"]["embedding"].T

# # Find the most relevant record
best_index = similarities.argmax()

best_record = knowledge_base["table"]["content"][best_index]
best_score = similarities[best_index]

# print("\nQuery:")
# print(query)

# print("\nTable Retrieval Result:")
# print(f"Similarity Score: {best_score:.4f}")
# print("Source: Table")
# print("Retrieved Record:")
# print(best_record)
#------------------------------------------------------------------------------
#check final answer
#------------------------------------------------------------------------------
context = best_record
prompt = f"""
Answer the user's question using only the provided context.

Question:
{query}

Context:
{context}

Give a concise answer.
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

# print("\nFinal Answer:")
# print(response["message"]["content"])


#------------------------------------------------------------------------------
#Test Image Retrieval
#------------------------------------------------------------------------------
# Step 15 — Test image retrieval

query = "What approval is required for international travel?"

# # Create CLIP embedding for the query
query_embedding = clip_model.encode(
    query,
    normalize_embeddings=True
)

# # Calculate similarity with the image
similarity = query_embedding @ knowledge_base["image"]["embedding"]

# print("\nQuery:")
# print(query)

# print("\nImage Retrieval Result:")
# print(f"Similarity Score: {similarity:.4f}")
# print("Source: Image")
# print("Retrieved Image:")
# print(knowledge_base["image"]["content"])


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Create a Unified Retrieval Function
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Unified multimodal retrieval

def multimodal_retrieve(query, top_k=5):

    results = []

    # -------------------------
    # Text retrieval
    # -------------------------
    query_embedding = text_model.encode(query)

    text_score = query_embedding @ knowledge_base["text"]["embedding"]

    results.append({
        "modality": "text",
        "score": float(text_score),
        "content": knowledge_base["text"]["content"]
    })

    # -------------------------
    # Table retrieval
    # -------------------------
    table_scores = query_embedding @ knowledge_base["table"]["embedding"].T

    for i, score in enumerate(table_scores):

        results.append({
            "modality": "table",
            "score": float(score),
            "content": knowledge_base["table"]["content"][i]
        })

    # -------------------------
    # Image retrieval
    # -------------------------
    image_query_embedding = clip_model.encode(
        query,
        normalize_embeddings=True
    )

    image_score = (
        image_query_embedding
        @ knowledge_base["image"]["embedding"]
    )

    results.append({
        "modality": "image",
        "score": float(image_score),
        "content": knowledge_base["image"]["content"]
    })

    # Sort by similarity
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]





# Step 17 — Test multimodal retrieval

# query = "What is the limit and approval requirement for international travel?"

# results = multimodal_retrieve(query)

# print("\nQuery:")
# print(query)

# print("\nRetrieved Sources:")

# for i, result in enumerate(results, start=1):

#     print(f"\n{i}. {result['modality'].upper()}")
#     print(f"Similarity Score: {result['score']:.4f}")
#     print(f"Content: {result['content']}")



#----------------------------------------------------------------------------------
#Build context from retrieved sources
# #----------------------------------------------------------------------------------
def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source Type: {result['modality']}\n"
            f"Content: {result['content']}"
        )

    return "\n\n".join(context_parts)

# context = build_context(results)

# print("\nRetrieved Context:")
# print(context)
# #----------------------------------------------------------------------------------
# #Generate the Final Answer
# #----------------------------------------------------------------------------------


# # Generate answer from retrieved context

def generate_answer(query, context):

    prompt = f"""
You are a helpful assistant.

Answer the user's question using only the retrieved context below.

If the context does not contain enough information to answer the question,
say that the information is not available in the retrieved sources.

Retrieved Context:
{context}

User Question:
{query}

Answer:
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

    return response["message"]["content"]


# answer = generate_answer(query, context)

# print("\nFinal Answer:")
# print(answer)









#Show Which Sources Were Used

# Step 20 — Display retrieved sources

# print("\nSources Used:")

# for i, result in enumerate(results, start=1):

#     print(
#         f"{i}. {result['modality'].capitalize()} "
#         f"(score: {result['score']:.4f})"
#     )










#----------------------------------------------------------
#Test Different Queries
#----------------------------------------------------------
#Now we can demonstrate why multimodal retrieval is useful.

#Query 1 — Text-focused
query = "How many days do employees have to submit travel expenses?"

results = multimodal_retrieve(query)

context = build_context(results)

answer = generate_answer(query, context)

print("\nQuestion:", query)
print("\nAnswer:", answer)

# #Query 2 — Table-focused
query = "What is the expense limit for international travel?"

results = multimodal_retrieve(query)

context = build_context(results)

answer = generate_answer(query, context)

print("\nQuestion:", query)
print("\nAnswer:", answer)

# #Query 3 — Image-focused
query = "Who needs to approve international travel?" #can also run this after removing the international travel part from table

results = multimodal_retrieve(query)

context = build_context(results)

answer = generate_answer(query, context)

print("\nQuestion:", query)
print("\nAnswer:", answer)
















#---------------------------------------------------------------------------------------------------------------
#Wrap Everything Into One RAG Function
#---------------------------------------------------------------------------------------------------------------
#  Complete multimodal RAG pipeline

# def multimodal_rag(query):

#     # Retrieve relevant sources
#     results = multimodal_retrieve(query)

#     # Build context from retrieved sources
#     context = build_context(results)

#     # Generate answer
#     answer = generate_answer(query, context)

#     return answer, results


# query = "What is the expense limit and approval requirement for international travel?"

# answer, results = multimodal_rag(query)

# print("\nQuestion:")
# print(query)

# print("\nAnswer:")
# print(answer)

# print("\nSources Used:")

# for i, result in enumerate(results, start=1):

#     print(
#         f"{i}. {result['modality'].capitalize()} "
#         f"(score: {result['score']:.4f})"
#     )