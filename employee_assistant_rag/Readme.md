Employee Assistant RAG System

A Retrieval-Augmented Generation (RAG) system built using LangChain, FAISS, LLaMA.cpp, FastAPI, and Streamlit to recommend suitable employees based on natural language queries.

Features

1. Natural Language Query Support

2. Vector Similarity Search with FAISS

3. Cosine Similarity based Retrieval

4. LLaMA LLM-powered Responses

5. REST API via FastAPI

6. Streamlit-based Frontend

7. Local model + embedding

Architecture Overview

graph TD
    User[👤 User (Client)]
    Streamlit[🖥️ Streamlit Frontend]
    FastAPI[⚙️ FastAPI Backend API]
    FAISS[🗂️ FAISS Vector Store (Embeddings)]
    LLaMA[🧠 LLaMA LLM (via llama.cpp)]

    User --> Streamlit
    Streamlit --> FastAPI
    FastAPI --> FAISS
    FastAPI --> LLaMA
    FAISS --> FastAPI
    LLaMA --> FastAPI


    User interacts via the Streamlit Frontend

    Streamlit Frontend sends queries to FastAPI Backend

    FastAPI Backend:

        Retrieves relevant document embeddings from FAISS Vector Store

        Feeds retrieved profiles as context to LLaMA LLM

        Returns generated responses back to Streamlit

    Streamlit Frontend displays final answer to the User

Setup & Installation

1️⃣ Clone the repo

    git clone <repo-url>
    cd employee-assistant-rag

2️⃣ Install dependencies
 
    pip install -r requirements.txt

3️⃣ Download LLaMA model
    
    Place llama-2-7b.Q4_0.gguf inside models/ directory.

4️⃣ Build FAISS Index
 
    python scripts/build_faiss_index_cosine.py


5️⃣ Run Backend API
 
    uvicorn backend.main:app --reload --port 8000

Open http://127.0.0.1:8000/docs

6️⃣ Run Frontend
 
    streamlit run frontend/app.py


Now test your workflow :

   - Go to your Streamlit app (localhost:8501)

   -  Enter a query like:
      "I need a machine learning engineer for a healthcare analytics project"

   -  Click Search

   You should see:
    - Retrieved employee summaries + similarity scores
    - Final LLaMA-generated paragraph recommendation


API Documentation

- POST /chat

    Request : {
                "query": "I need a machine learning engineer for a healthcare analytics project"
              } 

    Response: {
                "answer": "\n### Example 1:\n\nI recommend David Kim for this position. He has 5 years of experience in Python, TensorFlow, AWS, Machine Learning. Worked on projects like Image Recognition System, Healthcare Dashboard. Currently available.\n\nDavid is a highly skilled machine learning engineer with extensive experience in healthcare analytics. His expertise in image recognition and deep learning models has helped many organizations improve their patient care and outcomes. He is also proficient in Python, TensorFlow, AWS, and other relevant technologies.\n\nI would be happy to schedule a meeting or provide more details about David's qualifications and experience. Please let me know if you have any questions or concerns.\n\n### Example 2:\n\nI recommend Mia Chen for this position. She has 3 years of experience in Python, Pandas, Scikit-learn, Machine Learning. Worked on projects like Customer Segmentation, Healthcare Risk Analysis. Currently available.\n\nMia is a highly skilled machine learning engineer with extensive experience in healthcare analytics. Her expertise in customer segmentation and risk analysis has helped many organizations improve their patient care and outcomes. She is also proficient in Python, Pandas, Scikit-learn, and other relevant technologies.\n\nI would be happy to schedule a meeting or provide more details about Mia's qualifications and experience. Please let me know if you have any questions or concerns.\n"
              }

- GET /employees/search?query=...
    
    Description: Returns top 5 employee summaries with similarity scores.

    Response : {
                "query": "I need a machine learning engineer for a healthcare analytics project",
                "results": [
                    {
                    "summary": "Ava Thomas has 7 years of experience in Python, Docker, Kubernetes, Machine Learning. Worked on projects like Data Analytics Platform, Healthcare Dashboard, ML Model Deployment. Currently not available.",
                    "score": 0.61
                    },
                    {
                    "summary": "Isabella Lee has 4 years of experience in Python, AWS, FastAPI, Machine Learning. Worked on projects like Healthcare Dashboard, E-commerce Platform, Predictive Analytics for Healthcare. Currently not available.",
                    "score": 0.602
                    },
                    {
                    "summary": "David Kim has 5 years of experience in Python, TensorFlow, AWS, Machine Learning. Worked on projects like Image Recognition System, Healthcare Dashboard. Currently available.",
                    "score": 0.584
                    },
                    {
                    "summary": "Mia Chen has 3 years of experience in Python, Pandas, Scikit-learn, Machine Learning. Worked on projects like Customer Segmentation, Healthcare Risk Analysis. Currently available.",
                    "score": 0.557
                    },
                    {
                    "summary": "Sophia Zhang has 6 years of experience in Python, AWS, Docker, Machine Learning. Worked on projects like Data Analytics Platform, Predictive Maintenance System. Currently not available.",
                    "score": 0.529
                    }
                ]
              }


AI Development Process

- Created a synthetic realistic employee dataset.

- Built natural language summaries for each employee.

- Generated embeddings using HuggingFace all-MiniLM-L6-v2.

- Stored embeddings into FAISS (IndexFlatIP) after L2 normalization.

- Set up LangChain retrieval pipeline combining vector search + LLaMA LLM.

- Designed FastAPI endpoints for chat and employee search.

- Built a simple Streamlit frontend for interaction.

- Validated response quality and system performance.

AI Tools Used:

    ChatGPT (GPT-4o) via OpenAI for:

       - Code generation

       - Explaining LangChain, FAISS, and LLaMA concepts

       - Debugging error traces and resolving integration issues

       - Architecture design suggestions for the backend and retrieval chain setup

AI vs Handwritten Code Split

   - 70% AI-assisted code — Code snippets, module integrations, and debugging solutions suggested via ChatGPT.

   - 30% hand-written code — Final adjustments, JSON data management, Streamlit UI structuring, and error handling logic written manually after integrating AI-generated suggestions.

Interesting AI-Generated Optimizations

   - Suggested embedding normalization using faiss.normalize_L2() to ensure cosine similarity retrieval correctness.

   - Recommended separating GET /employees/search for lightweight similarity scoring without LLaMA inference.


Future Improvements:

    Model & Embedding Improvements

        - Switch to more advanced embedding models like all-MiniLM-L12-v2, BAAI/bge-base-en-v1.5, or intfloat/e5-large-v2 for better semantic search accuracy.

        - Experiment with domain-specific embeddings fine-tuned for HR and recruitment text to improve relevancy in candidate retrieval.

        - Integrate larger, more powerful models (e.g., LLaMA 3, Mistral 7B) for improved reasoning and language fluency.

        - Optimize FAISS index search parameters and experiment with other similarity metrics (e.g., Hierarchical Navigable Small World (HNSW)) for faster retrieval on large datasets.