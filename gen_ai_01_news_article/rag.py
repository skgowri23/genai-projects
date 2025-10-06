import requests
from bs4 import BeautifulSoup
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Global constants
VECTORSTORE_DIR = Path("./resources/vectorstore")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name="english_news",
    embedding_function=ef,
    persist_directory=str(VECTORSTORE_DIR)
)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=api_key)


def scrape_article(url: str):
    """Fetch article text + metadata from Hindu article URL"""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "lxml")

    # Extract article body
    article_body = soup.find("div", {"itemprop": "articleBody"})
    if not article_body:
        return None

    article_text = " ".join([p.get_text(strip=True) for p in article_body.find_all("p")])

    # Extract metadata
    title = soup.find("title").get_text(strip=True) if soup.find("title") else "Untitled"
    published_date = None
    published_tag = soup.find("meta", {"property": "article:published_time"})
    if published_tag:
        published_date = published_tag.get("content")

    return {
        "text": article_text,
        "title": title,
        "date": published_date,
        "url": url,
    }


def ingest_article(url: str):
    """Scrape + split + store article into Chroma with metadata"""
    article = scrape_article(url)
    if not article:
        return False, "No article content found"

    doc = Document(
        page_content=article["text"],
        metadata={
            "source": article["url"],
            "date": article["date"],
            "title": article["title"],
        },
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([doc])

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    return True, f"Stored {len(docs)} chunks for: {article['title']}"


def ask_question(query: str, k: int = 3):
    """Retrieve context and get answer from LLM, include metadata"""
    results = vector_store.similarity_search(query, k=k)

    context = "\n".join([r.page_content for r in results])
    prompt = f"""
    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = llm.invoke(prompt)

    # Deduplicate by source URL
    seen_urls = set()
    sources = []
    for r in results:
        url = r.metadata.get("source", "")
        if url not in seen_urls:   # only keep first occurrence
            seen_urls.add(url)
            src = f"- {r.metadata.get('title', 'Untitled')} ({r.metadata.get('date', 'Unknown')})\n  {url}"
            sources.append(src)

    return response.content, sources
