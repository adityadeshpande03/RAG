from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from typing import List, Optional
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from chromadb import PersistentClient
import json

app = FastAPI()

GROQ_API_KEY = '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'

@app.post("/ask_question")
async def ask_question(
    question: str = Form(...),
    urls: Optional[List[str]] = Body(default=[]),  # Changed to default=[]
    files: Optional[List[UploadFile]] = File(default=[])  # Made files optional with default
):
    try:
        # Validate inputs
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not urls and not files:
            raise HTTPException(status_code=400, detail="Please provide either URLs or files")

        # Setup the RAG pipeline
        vectorstore, rag_chain = await setup_rag_pipeline(urls, files)
        
        # Get relevant documents
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Limit number of documents
        )
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")
            
        docs_content = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer
        result = rag_chain.invoke({
            "context": docs_content,
            "question": question
        })

        # Validate result
        if not result or not isinstance(result, str):
            raise HTTPException(status_code=500, detail="Failed to generate a valid response")

        return {"answer": result}

    except Exception as e:
        # Log the error (you should add proper logging)
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def setup_rag_pipeline(urls: List[str], uploaded_files: List[UploadFile]):
    try:
        all_docs = []
        
        # Process URLs
        if urls:
            for url in urls:
                try:
                    loader = WebBaseLoader(url.strip())
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading URL {url}: {str(e)}")
                    continue
        
        # Process files
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join(temp_dir, uploaded_file.filename)
                    content = await uploaded_file.read()  # Need to make this function async
                    with open(file_path, "wb") as f:
                        f.write(content)
                    
                    file_extension = uploaded_file.filename.split('.')[-1].lower()
                    docs = await load_document(file_path, file_extension)
                    if docs:
                        all_docs.extend(docs)
                    
                    # Clean up temp file
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error processing file {uploaded_file.filename}: {str(e)}")
                    continue
            
            # Clean up temp directory
            os.rmdir(temp_dir)

        if not all_docs:
            raise ValueError("No documents were successfully loaded")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=30  # Added some overlap
        )
        doc_splits = text_splitter.split_documents(all_docs)

        # Initialize vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2",
            model_kwargs={'device': 'cpu'}  # Explicitly set device
        )
        
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            persist_directory=tempfile.mkdtemp()  # Temporary storage
        )

        # Initialize LLM and chain
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model='llama-3.3-70b-versatile',
            temperature=0,
            max_tokens=1000  # Add token limit
        )
        
        rag_prompt = hub.pull("rlm/rag-prompt")
        rag_chain = rag_prompt | llm | StrOutputParser()

        return vectorstore, rag_chain

    except Exception as e:
        raise ValueError(f"Failed to setup RAG pipeline: {str(e)}")

async def load_document(file_path: str, file_type: str):
    supported_types = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "json": lambda path: JSONLoader(file_path=path, jq_schema='.[]', text_content=False)
    }
    
    if file_type not in supported_types:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    try:
        loader = supported_types[file_type](file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        return []
