import os
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv

from rag import scrape_article
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq

# ---------- Setup ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="News Q&A", page_icon="bird.png")
st.markdown("### 📰 Ask Questions About a News Article")

# ---------- Helpers ----------
def new_vectorstore():
    ef = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    return Chroma(collection_name="temp_session", embedding_function=ef, persist_directory=None)

def build_vectorstore(text: str, url: str):
    vector_store = new_vectorstore()  # 🔑 always a fresh instance
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text, metadata={"source": url})])
    vector_store.add_documents(docs, ids=[str(uuid4()) for _ in docs])
    return vector_store, len(docs)

def reset_session():
    st.session_state.clear()   # 🔑 nuke everything
    st.success("Session reset.")
    st.rerun()

# ---------- Top controls ----------
col_left, col_right = st.columns([4, 1])
with col_left:
    url = st.text_input("Enter Article URL:", key="article_url")
with col_right:
    if st.button("Reset"):
        reset_session()

if url and st.button("Load Article"):
    url = url.strip().strip('"').strip("'")  # tolerate quotes/spaces

    # Always flush and rebuild for new load
    with st.spinner("Scraping and embedding the article..."):
        article = scrape_article(url)
        if not article or not article.get("text"):
            st.error("❌ Could not extract text from this URL.")
            st.stop()

        vector_store, total = build_vectorstore(article["text"], url)

    st.session_state["vector_store"] = vector_store
    st.session_state["loaded_url"] = url
    st.success(f"✅ Loaded article and created {total} chunks.")

# ---------- Q&A ----------
query = st.text_input("Ask a question about the loaded article:", key="query")

if query and "vector_store" in st.session_state:
    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            results = st.session_state["vector_store"].similarity_search(query, k=4)
            if not results:
                st.info("No relevant content found in this article.")
            else:
                context = "\n".join(r.page_content for r in results)
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=GROQ_API_KEY)
                prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
                response = llm.invoke(prompt)

                st.subheader("🤖 Answer")
                st.write(response.content)

                st.subheader("📌 Sources")
                seen = set()
                for r in results:
                    s = r.metadata.get("source", "")
                    if s and s not in seen:
                        seen.add(s)
                        st.markdown(f"- [{s}]({s})")
