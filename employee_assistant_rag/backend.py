from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Initialize FastAPI app
app = FastAPI(
    title="Employee Assistant API",
    description="LangChain-powered RAG assistant for employee recommendations.",
    version="1.0.0"
)

# 📝 Request schema for POST /chat
class ChatRequest(BaseModel):
    query: str

# 📦 Load LLM and Vector Store on app startup
hf_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    "/path/to/faiss_index_cosine",
    hf_embeddings,
    allow_dangerous_deserialization=True
)

llm = LlamaCpp(
    model_path = "/path/to/models/llama-2-7b.Q4_0.gguf",
    n_ctx=4096,
    max_tokens=1024,
    temperature=0.0
)

template = """
You are an HR assistant. The user’s request: {input}

Here are candidate profiles:
{context}

Write a clear, polite, natural-sounding paragraph summarizing your recommendation.  
For each candidate, mention:
- Their name  
- Relevant years of experience  
- Notable projects or skills relevant to the request  
- Their current availability  

End your response by politely offering to schedule a meeting or provide more details.
"""

prompt = PromptTemplate(
    input_variables=["input", "context"],
    template=template
)

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    combine_docs_chain=combine_docs_chain
)

# POST /chat — Chat RAG endpoint
@app.post("/chat")
def chat_query(request: ChatRequest):
    try:
        result = rag_chain.invoke({"input": request.query})
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")

# GET /employees/search — Search employees with similarity scores
@app.get("/employees/search")
def search_employees(query: str = Query(...)):
    try:
        results = vectorstore.similarity_search_with_score(query, k=5)
        employees = []
        for doc, score in results:
            employees.append({
                "summary": doc.page_content,
                "score": round(float(score), 3)
            })
        return {"query": query, "results": employees}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search employees: {e}")
