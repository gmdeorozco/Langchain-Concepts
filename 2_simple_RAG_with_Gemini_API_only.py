import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Step 1: Initialize models
# Embeddings are obtained directly via genai.embed_content
# LLM is initialized as GenerativeModel
llm = genai.GenerativeModel("gemini-2.5-flash-lite")

# Step 2: Create a text corpus and generate embeddings
documents = ["harrison worked at kensho"]

# Embed the document
document_embeddings = genai.embed_content(
    model="models/embedding-001",
    content=documents[0]
)

document_vector = np.array([document_embeddings["embedding"]], dtype="float32")

# Step 3: Build the FAISS index
dimension = len(document_vector[0])
index = faiss.IndexFlatL2(dimension)
index.add(document_vector)

# Step 4: Process the user query
query = "where did harrison work?"

query_embedding_response = genai.embed_content(
    model="models/embedding-001",
    content=query
)
query_vector = np.array([query_embedding_response["embedding"]], dtype="float32")

# Step 5: Perform similarity search
k = 1
distances, indices = index.search(query_vector, k)
retrieved_context = documents[indices[0][0]]

# Step 6: Construct the prompt
template = f"""
Answer the question based only on the following context, give a complete, friendly answer:
Context: {retrieved_context}
Question: {query}
"""

# Step 7: Generate the response
try:
    response = llm.generate_content(template)
    print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")
