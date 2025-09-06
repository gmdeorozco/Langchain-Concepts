import openai
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Step 1: Initialize OpenAI Client and Models ---
# The OpenAI client is a single entry point for all API calls.
openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# We will use the 'text-embedding-3-small' model for embeddings
# and a conversational model like 'gpt-4o-mini' for text generation.
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# --- Step 2: Create a text corpus and generate embeddings ---
documents = ["Harrison worked at Kensho."]

# Get the embedding for the document using the OpenAI client
try:
    document_embeddings = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=documents[0]
    )
    # The embedding is in a data list, so we need to extract it.
    document_vector = np.array([document_embeddings.data[0].embedding], dtype="float32")
except Exception as e:
    print(f"An error occurred while creating document embeddings: {e}")
    exit()

# --- Step 3: Build the FAISS index ---
dimension = len(document_vector[0])
index = faiss.IndexFlatL2(dimension)
index.add(document_vector)

# --- Step 4: Process the user query ---
query = "Where did Harrison work?"

# Get the embedding for the query
try:
    query_embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vector = np.array([query_embedding_response.data[0].embedding], dtype="float32")
except Exception as e:
    print(f"An error occurred while creating query embedding: {e}")
    exit()

# --- Step 5: Perform similarity search ---
k = 1
distances, indices = index.search(query_vector, k)
retrieved_context = documents[indices[0][0]]

# --- Step 6: Construct the prompt for the LLM ---
# Use the ChatCompletions API format with roles.
prompt_messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer the user's question based only on the provided context. If the answer is not in the context, say so."},
    {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {query}"}
]

# --- Step 7: Generate the final response ---
try:
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=prompt_messages,
        temperature=0.0
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred during content generation: {e}")