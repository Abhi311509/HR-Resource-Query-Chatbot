import os
import json
import numpy as np
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.docstore import InMemoryDocstore

# Load employee JSON
with open("data.json", "r") as f:
    employees = json.load(f)["employees"]

# Initialize embedder (CPU)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Convert each employee record into a human-readable text summary
docs = []
for emp in employees:
    summary = (
        f"{emp['name']} has {emp['experience_years']} years of experience in "
        f"{', '.join(emp['skills'])}. Worked on projects like {', '.join(emp['projects'])}. "
        f"Currently {emp['availability']}."
    )
    docs.append(
        Document(
            page_content=summary,
            metadata={
                "id": emp["id"],
                "name": emp["name"],
                "experience_years": emp["experience_years"],
                "skills": emp["skills"],
                "projects": emp["projects"],
                "availability": emp["availability"]
            }
        )
    )

# Compute embeddings for documents
texts = [doc.page_content for doc in docs]
embeddings = embedding_model.embed_documents(texts)
embeddings = np.array(embeddings).astype("float32")

# Normalize embeddings to unit length (for cosine similarity)
faiss.normalize_L2(embeddings)

# Create FAISS IndexFlatIP (for cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Create docstore and index-to-docstore mapping
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
index_to_docstore_id = {i: str(i) for i in range(len(docs))}

# Wrap into LangChain FAISS Vectorstore
vectorstore = FAISS(embedding_model, index, docstore, index_to_docstore_id)

# Save index to disk
index_path = "/home/examroom/CB/faiss_index_cosine"
os.makedirs(index_path, exist_ok=True)
vectorstore.save_local(index_path)

# Log final status
print(f"FAISS Cosine Similarity index built and saved to: {index_path}")
print(f"New FAISS index built with {len(docs)} valid documents.")
print(vectorstore.index)
