# prompts/templates.py
# This file defines prompt templates for the contract evaluation tool.
# These templates are used in chains/analysis.py and chains/rag.py.
# Integrates with app.py by providing reusable prompts for LLM calls.

from langchain_core.prompts import PromptTemplate

# Combined Analysis Prompt Template
# Used for clause extraction, risk scoring (1-10), flagging ambiguities, and summarization in a single chain.
# Improvements: Added Zero-Shot Chain-of-Thought (COT) instruction, Few-Shot example, and refined JSON schema with overall_risk_score as null placeholder.
# Updated: Added "clause_name" to each clause object for better weight matching in analysis.py.
ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["contract_text"],
    template="""
    You are an expert legal analyst working for Hari and Winston Associates LLC (HWA), specializing in data analytics and consulting contracts. 
    Before outputting the JSON, internally reason step-by-step (Zero-Shot Chain-of-Thought): 
    - Identify key clauses.
    - Evaluate risks considering HWA's context (e.g., financial liabilities, IP protection, service delivery timelines).
    - Weigh risks on a 1-10 scale (1=low, 10=high) based on potential impact.
    - Note ambiguities and flags.

    Analyze the following contract text:

    {contract_text}

    Perform the following tasks:
    1. **Extract Key Clauses**: Identify and list all major clauses, including parties, terms, payment, termination, liabilities, and any non-standard clauses. For each, provide a concise "clause_name" (e.g., "Payment Terms").
    2. **Risk Scoring and Flagging**: For each extracted clause, assign a risk score from 1 to 10 (1 = low risk, 10 = high risk) based on potential legal, financial, or operational risks. Flag any high-risk (score > 7) or ambiguous sections with explanations.
    3. **Highlight Ambiguities**: Identify and highlight any vague, unclear, or potentially disputable language in the contract.
    4. **Summarization**: Provide a concise summary of the entire contract, including key obligations, risks, and recommendations.

    Few-Shot Example:
    Input Clause: "The supplier shall deliver within 30 days."
    Output Object: 
    {{
      "clause_name": "Delivery Timeline",
      "clause_text": "The supplier shall deliver within 30 days.",
      "risk_score": 3,
      "risk_explanation": "Standard timeline with low risk of delay for HWA's operations.",
      "is_ambiguous": false,
      "ambiguity_details": ""
    }}

    Output strictly in this JSON schema (no extra text):
    {{
      "summary": "string (concise contract summary)",
      "clauses": [array of objects like the example above],
      "flags": ["array of strings for high-risk flags, e.g., 'High risk in payment terms'"],
      "overall_risk_score": null  // Placeholder; average will be calculated in code
    }}
    Ensure the output is valid JSON.
    """
)

# RAG Q&A Prompt Template
# Used for the retrieval-augmented generation in the chatbot.
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant specializing in contract analysis. Use the following retrieved context from contracts to answer the user's question accurately.

    Context:
    {context}

    Question: {question}

    Provide a clear, concise answer based on the context. If the question cannot be answered from the context, state that and suggest rephrasing. Do not add external knowledge.
    """
)

# Optional: Chat History Integration for RAG (if needed for conversational memory)
CONVERSATIONAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
    You are a helpful assistant specializing in contract analysis. Use the following chat history and retrieved context to answer the user's question.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {question}

    Provide a relevant answer, maintaining conversation flow. Base responses on the provided context and history.
    """
)
