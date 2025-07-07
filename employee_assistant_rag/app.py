import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("💼 HR Assistant Chatbot")

# User input
user_query = st.text_input("Enter your query:")

if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Contacting assistant..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": user_query}
                )
                if response.status_code == 200:
                    result = response.json()

                    st.subheader("📑 Final Recommendation:")
                    st.write(result["answer"])

                else:
                    st.error(f"Error: {response.status_code} — {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
