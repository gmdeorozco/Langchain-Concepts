from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 
template = """Answer the question based only on the following context, give a complete, friendly answer:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Gemini's embedding model
)

query = "where did harrison work?"
docs = vectorstore.similarity_search(query, top_k=1)

retriever = vectorstore.as_retriever()
docs = retriever.invoke(query, top_k=1)

chain = (
    {
    "context": retriever,
    "question": RunnablePassthrough()
    } 
    | prompt
    | llm
    | StrOutputParser()
    )

response = chain.invoke(query)
print(response)