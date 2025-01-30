from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from chromadb import PersistentClient
import os
import tempfile
import re
import uvicorn
from bs4 import BeautifulSoup  # Add this import


# Set User-Agent
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# API Keys
GROQ_API_KEY = "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"
TAVILY_API_KEY = "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"

# Initialize FastAPI app
app = FastAPI(title="Corrective - RAG System API")

# Define request/response models
class QueryRequest(BaseModel):
    question: str
    urls: Optional[List[str]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is FastAPI?",
                "urls": ["https://fastapi.tiangolo.com/"]
            }
        }

class QueryResponse(BaseModel):
    response: str
    sources: List[str]

# Define the RAG prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the question. If you're unsure or the context doesn't contain relevant information, say so."),
    ("user", "Context: {context}\n\nQuestion: {question}")
])

class RAGSystem:
    def __init__(self):
        # Create a temporary directory for Chroma
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250,
                chunk_overlap=0
            )
        except ImportError:
            # Fallback to basic character splitting if tiktoken is not available
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,
                chunk_overlap=0
            )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model='llama-3.3-70b-versatile',
            temperature=0
        )
        # Set Tavily API key in environment variable
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        self.web_search_tool = TavilySearchResults()
        self.vectorstore = None

    def load_documents(self, urls: List[str]) -> List[Document]:
        """Load and split documents from URLs."""
        if not urls:
            return []
            
        try:
            docs = []
            for url in urls:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading documents: {str(e)}")

    def setup_vectorstore(self, documents: List[Document]) -> bool:
        """Initialize the vector store with documents."""
        if not documents:
            return False
            
        try:
            # Initialize PersistentClient with the temporary directory
            chroma_client = PersistentClient(path=self.temp_dir)
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=chroma_client,
                collection_name='rag-chroma'
            )
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error setting up vector store: {str(e)}")

    def web_search(self, question: str) -> List[Document]:
        """Perform web search and return results as documents."""
        try:
            results = self.web_search_tool.invoke({"query": question})
            if results:
                return [Document(page_content=result["content"], metadata={"source": "web_search"}) 
                       for result in results]
            return []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during web search: {str(e)}")

    def process_query(self, question: str, urls: List[str] = None) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        try:
            docs = []
            
            # Load and process documents if URLs provided
            if urls:
                documents = self.load_documents(urls)
                if documents:
                    self.setup_vectorstore(documents)
                    docs.extend(self.vectorstore.as_retriever().get_relevant_documents(question))
            
            # Add web search results
            web_docs = self.web_search(question)
            docs.extend(web_docs)
            
            if not docs:
                raise HTTPException(status_code=404, detail="No relevant information found.")
            
            # Generate response
            context = "\n".join([doc.page_content for doc in docs])
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Extract sources
            sources = [doc.metadata.get("source", "unknown") for doc in docs]
            
            return {
                "response": response,
                "sources": sources
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Initialize RAG system
rag_system = RAGSystem()

# Define endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query with optional URLs for context.
    
    - question: The question to answer
    - urls: Optional list of URLs to provide additional context
    """
    result = rag_system.process_query(request.question, request.urls)
    return QueryResponse(
        response=result["response"],
        sources=result["sources"]
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)