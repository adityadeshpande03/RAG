import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, JSONLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from typing import List, Literal, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain import hub
from pprint import pprint
from langchain_core.runnables import RunnablePassthrough

# FastAPI instance
app = FastAPI()

# Set up API keys
GROQ_API_KEY = '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'
TAVILY_API_KEY = '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'
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Create a temporary directory for file uploads
UPLOAD_DIR = Path(tempfile.gettempdir()) / "fastapi_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Embedding and document setup
embd = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2')
web_search_tool = TavilySearchResults(k=3)

# Setup the LLM (Groq)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model='llama-3.3-70b-versatile',
    temperature=0
)

async def save_upload_file(upload_file: UploadFile) -> Path:
    """Safely save an uploaded file and return its path."""
    try:
        file_path = UPLOAD_DIR / f"{hash(upload_file.filename)}_{upload_file.filename}"
        with open(file_path, "wb") as f:
            while content := await upload_file.read(1024 * 1024):
                f.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file {upload_file.filename}: {str(e)}"
        )

def load_documents(file_paths: List[Path], urls: List[str]) -> List[Document]:
    """Load documents from files and URLs with better error handling."""
    docs = []
    
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading URL {url}: {str(e)}")
    
    for file_path in file_paths:
        try:
            if str(file_path).lower().endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
            elif str(file_path).lower().endswith(".docx"):
                loader = Docx2txtLoader(str(file_path))
            elif str(file_path).lower().endswith(".json"):
                loader = JSONLoader(str(file_path))
            elif str(file_path).lower().endswith(".txt"):
                loader = TextLoader(str(file_path))
            else:
                print(f"Skipping unsupported file type: {file_path}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
    
    return docs

# Text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)

# Define RAG components
def format_docs(docs: List[Document]) -> str:
    """Format documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# RAG prompt template
template = """Answer the following question based on the provided context.

Context: {context}

Question: {question}

Answer: """

prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
def create_chain():
    return (
        {
            "context": lambda x: format_docs(x["documents"]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = create_chain()

# Define GraphState
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: Optional[str]

def retrieve(state: GraphState) -> GraphState:
    """Retrieve relevant documents."""
    try:
        print(f"Retrieving documents for question: {state['question']}")
        documents = retriever.get_relevant_documents(state["question"])
        print(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "question": state["question"], "generation": None}
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        raise e

def generate(state: GraphState) -> GraphState:
    """Generate an answer based on the question and retrieved documents."""
    try:
        print(f"Generating answer for question: {state['question']}")
        print(f"Using {len(state['documents'])} documents")
        
        # Ensure documents is a list
        docs = state["documents"]
        if isinstance(docs, Document):
            docs = [docs]
        
        generation = rag_chain.invoke({
            "documents": docs,
            "question": state["question"]
        })
        
        print(f"Generated answer: {generation}")
        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": generation
        }
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        raise e

def web_search(state: GraphState) -> GraphState:
    """Perform web search for the question."""
    try:
        print(f"Performing web search for: {state['question']}")
        results = web_search_tool.invoke({"query": state["question"]})
        documents = [Document(page_content=result["content"]) for result in results]
        print(f"Found {len(documents)} web results")
        return {
            "documents": documents,
            "question": state["question"],
            "generation": None
        }
    except Exception as e:
        print(f"Error in web_search: {str(e)}")
        raise e

def route_question(state: GraphState) -> str:
    """Route the question to appropriate source."""
    try:
        print(f"Routing question: {state['question']}")
        # Simple routing logic - if we have documents, use them; otherwise web search
        if hasattr(state, 'documents') and state['documents']:
            return "vectorstore"
        return "web_search"
    except Exception as e:
        print(f"Error in route_question: {str(e)}")
        raise e

# Define workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Add edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile workflow
app_workflow = workflow.compile()

# Global retriever
retriever = None

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    urls: List[str] = Form([]),
    files: List[UploadFile] = File([])
):
    global retriever
    saved_files: List[Path] = []

    try:
        print(f"\nProcessing new question: {question}")
        
        # Save uploaded files
        for file in files:
            file_path = await save_upload_file(file)
            saved_files.append(file_path)
            print(f"Saved file: {file_path}")

        # Load documents
        docs = load_documents(saved_files, urls)
        if not docs and not urls:
            print("No documents provided, will use web search")
        else:
            print(f"Loaded {len(docs)} documents")
            # Split documents
            doc_splits = text_splitter.split_documents(docs)
            print(f"Created {len(doc_splits)} document splits")

            # Create vectorstore and retriever
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=embd,
            )
            retriever = vectorstore.as_retriever()

        # Process workflow
        inputs = {
            "question": question,
            "documents": docs if docs else [],
            "generation": None
        }
        
        print("Starting workflow processing")
        last_output = None
        for output in app_workflow.stream(inputs):
            print(f"Workflow step output: {output}")
            last_output = output
            
            # Check if we have a generation in the nested structure
            if isinstance(last_output, dict) and 'generate' in last_output:
                generation = last_output['generate'].get('generation')
                if generation:
                    return {"answer": generation}

        # Final check in case the last output contains the answer
        if last_output:
            if isinstance(last_output, dict):
                # Try to find generation in nested structure
                if 'generate' in last_output and last_output['generate'].get('generation'):
                    return {"answer": last_output['generate']['generation']}
                elif 'generation' in last_output:
                    return {"answer": last_output['generation']}
        
        raise HTTPException(
            status_code=500,
            detail="No answer was generated"
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the request: {str(e)}"
        )
    finally:
        # Clean up files
        for file_path in saved_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)