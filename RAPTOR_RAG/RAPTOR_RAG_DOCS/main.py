# requirements.txt
"""
fastapi
python-multipart
uvicorn
pydantic
numpy
pandas
umap-learn
beautifulsoup4
scikit-learn
tiktoken
langchain
langchain-community
langchain-groq
langchain-huggingface
chromadb
sentence-transformers
python-docx
pypdf
"""

# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from umap.umap_ import UMAP
from bs4 import BeautifulSoup as Soup
from sklearn.mixture import GaussianMixture
import tiktoken
import json
from docx import Document
from pypdf import PdfReader
import io

from langchain_community.document_loaders import (
    RecursiveUrlLoader,
    TextLoader,
    JSONLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

app = FastAPI()

# Constants
RANDOM_SEED = 224
GROQ_API_KEY = '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'  # Replace with your actual key
MAX_TOKENS = 4000
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.json'}

# Initialize models
embd = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = ChatGroq(
    model='deepseek-r1-distill-llama-70b',
    temperature=0,
    api_key=GROQ_API_KEY,
    max_tokens=MAX_TOKENS
)

# Text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=lambda x: num_tokens_from_string(x, "cl100k_base"),
    separators=["\n\n", "\n", " ", ""]
)

# Pydantic model for response
class ProcessingResponse(BaseModel):
    answer: str
    source_count: int
    
# Helper functions
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Calculate the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    """Perform global clustering on embeddings using UMAP."""
    if n_neighbors is None:
        n_neighbors = min(int((len(embeddings) - 1) ** 0.5), len(embeddings) - 1)
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    """Perform local clustering on embeddings using UMAP."""
    num_neighbors = min(num_neighbors, len(embeddings) - 1)
    return UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50) -> int:
    """Determine optimal number of clusters using BIC criterion."""
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=RANDOM_SEED)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    
    return n_clusters[np.argmin(bics)]

def GMM_cluster(embeddings: np.ndarray, threshold: float) -> tuple:
    """Perform Gaussian Mixture Model clustering."""
    n_clusters = max(1, min(get_optimal_clusters(embeddings), len(embeddings) - 1))
    gm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
    """Perform hierarchical clustering on embeddings."""
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_mask]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)

        for j in range(n_local_clusters):
            local_cluster_mask = np.array([j in lc for lc in local_clusters])
            local_cluster_embeddings_ = global_cluster_embeddings_[local_cluster_mask]
            
            indices = []
            for lcemb in local_cluster_embeddings_:
                idx = np.where((embeddings == lcemb).all(axis=1))[0]
                indices.extend(idx)
            
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    return all_local_clusters

def embed(texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    text_embeddings = embd.embed_documents(texts)
    return np.array(text_embeddings)

def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    """Embed and cluster texts, returning a DataFrame with results."""
    text_embeddings_np = embed(texts)
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
    return pd.DataFrame({
        "text": texts,
        "embd": list(text_embeddings_np),
        "cluster": cluster_labels
    })

def chunk_text(text: str) -> List[str]:
    """Split text into manageable chunks."""
    return text_splitter.split_text(text)

def summarize_chunk(chunk: str, template: str) -> str:
    """Summarize a chunk of text using the LLM."""
    try:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"context": chunk})
    except Exception as e:
        print(f"Error summarizing chunk: {str(e)}")
        return ""

def embed_cluster_summarize_texts(texts: List[str], level: int) -> tuple:
    """Embed, cluster, and summarize texts at a given level."""
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    df_clusters = embed_cluster_texts(all_chunks)
    expanded_list = []

    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append({
                "text": row["text"],
                "embd": row["embd"],
                "cluster": cluster
            })

    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df["cluster"].unique()

    template = """Here is a sub-set of documentation. Give a detailed summary of the documentation provided.

    Documentation:
    {context}
    """

    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        cluster_text = "\n--- --- \n".join(df_cluster["text"].tolist())
        summary = summarize_chunk(cluster_text, template)
        summaries.append(summary)

    df_summary = pd.DataFrame({
        "summaries": summaries,
        "level": [level] * len(summaries),
        "cluster": list(all_clusters),
    })

    return df_clusters, df_summary

def recursive_embed_cluster_summarize(texts: List[str], level: int = 1, n_levels: int = 3) -> Dict[int, tuple]:
    """Recursively embed, cluster, and summarize texts across multiple levels."""
    results = {}
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(new_texts, level + 1, n_levels)
        results.update(next_level_results)

    return results

async def process_file(file: UploadFile) -> List[str]:
    """Process uploaded files and extract text content."""
    content = await file.read()
    file_extension = file.filename.lower()[file.filename.rfind('.'):]
    
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        if file_extension == '.txt':
            text = content.decode('utf-8')
            return [text]
            
        elif file_extension == '.pdf':
            pdf = PdfReader(io.BytesIO(content))
            return [page.extract_text() for page in pdf.pages]
            
        elif file_extension == '.docx':
            doc = Document(io.BytesIO(content))
            return ['\n'.join([paragraph.text for paragraph in doc.paragraphs])]
            
        elif file_extension == '.json':
            json_data = json.loads(content)
            if isinstance(json_data, str):
                return [json_data]
            elif isinstance(json_data, list):
                return [str(item) for item in json_data]
            else:
                return [json.dumps(json_data)]
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

async def process_url(url: str, max_depth: int = 20) -> List[str]:
    """Process URLs and extract text content."""
    try:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()
        return [doc.page_content for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing URL {url}: {str(e)}")

async def process_documents(docs_texts: List[str], question: str) -> ProcessingResponse:
    """Process documents and generate a response to the question."""
    results = recursive_embed_cluster_summarize(docs_texts, level=1, n_levels=3)

    all_texts = docs_texts.copy()
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)

    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    return ProcessingResponse(
        answer=answer,
        source_count=len(docs_texts)
    )

# Unified API Endpoint
@app.post("/process/", response_model=ProcessingResponse)
async def process_content(
    question: str = Form(...),
    files: List[UploadFile] = File(None),
    urls: str = Form(None)
):
    """
    Process both files and URLs in a single endpoint.
    
    Args:
        question: The question to answer about the content
        files: Optional list of files to process
        urls: Optional comma-separated list of URLs to process
    """
    try:
        docs_texts = []

        # Process files if provided
        if files:
            for file in files:
                file_texts = await process_file(file)
                for text in file_texts:
                    chunks = chunk_text(text)
                    docs_texts.extend(chunks)

        # Process URLs if provided
        if urls:
            url_list = [url.strip() for url in urls.split(',')]
            for url in url_list:
                url_texts = await process_url(url)
                for text in url_texts:
                    chunks = chunk_text(text)
                    docs_texts.extend(chunks)

        if not docs_texts:
            raise HTTPException(
                status_code=400,
                detail="No content provided. Please provide either files or URLs."
            )

        return await process_documents(docs_texts, question)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)