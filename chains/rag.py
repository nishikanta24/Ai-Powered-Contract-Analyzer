import os
import json
from dotenv import load_dotenv
import logging

from langchain_groq import ChatGroq

load_dotenv()

class ContractRAGChain:
    def __init__(self, vector_store):
        """Initialize RAG Chain with vector store and Groq LLM"""
        try:
            self.llm = ChatGroq(
                model=os.getenv("GROQ_MODEL"),
                temperature=0.0,
                max_tokens=1000,
                api_key=os.getenv("GROQ_API_KEY")
            )
            self.vector_store = vector_store
            self.retriever = vector_store.as_retriever()
        except Exception as e:
            logging.error(f"Error initializing ContractRAGChain: {e}")
            raise e

    def invoke(self, question):
        """Use retriever + LLM to answer questions based on contract context"""
        try:
            docs = self.retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"Based on this contract:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            response = self.llm.invoke(prompt)
            
            return response.content
        except Exception as e:
            logging.error(f"Error during RAG chain invocation: {e}")
            raise e
