import streamlit as st
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import os
import tempfile
from chromadb.config import Settings
from chromadb import PersistentClient
import json

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.processed_files = set()

# Set page config
st.set_page_config(page_title="RAG Application", layout="wide")

# Create a temporary directory for Chroma and uploaded files
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.upload_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(st.session_state.temp_dir, "chroma"), exist_ok=True)

# Your Groq API key
GROQ_API_KEY = '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'  # Replace with your actual API key

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def load_document(file_path, file_type):
    try:
        if file_type == "pdf":
            return PyPDFLoader(file_path).load()
        elif file_type == "docx":
            return Docx2txtLoader(file_path).load()
        elif file_type == "txt":
            return TextLoader(file_path).load()
        elif file_type == "json":
            return JSONLoader(
                file_path=file_path,
                jq_schema='.[]',
                text_content=False
            ).load()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        st.error(f"Error loading document {file_path}: {str(e)}")
        return []

def initialize_components():
    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile',
        temperature=0
    )
    
    # Initialize prompts
    retrieval_grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts."),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ])
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an answer addresses / resolves a question"),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ])
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval."),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ])
    
    return llm, retrieval_grade_prompt, hallucination_prompt, answer_prompt, rewrite_prompt

def setup_rag_pipeline(urls, uploaded_files):
    try:
        # Process URLs
        url_docs = []
        if urls:
            for url in urls:
                if url.strip():
                    try:
                        loader = WebBaseLoader(url.strip())
                        docs = loader.load()
                        url_docs.extend(docs)
                    except Exception as e:
                        st.warning(f"Failed to load URL {url}: {str(e)}")
                        continue
        
        # Process uploaded files
        file_docs = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                # Save the uploaded file
                file_path = os.path.join(st.session_state.upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Determine file type and load document
                file_extension = uploaded_file.name.split('.')[-1].lower()
                docs = load_document(file_path, file_extension)
                file_docs.extend(docs)
                st.session_state.processed_files.add(uploaded_file.name)
        
        # Combine all documents
        all_docs = url_docs + file_docs
        
        if not all_docs:
            st.warning("No documents were successfully loaded. Please check your inputs.")
            return None, None, None, None, None, None, None
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(all_docs)
        
        # Initialize Chroma with persistent client
        chroma_client = PersistentClient(
            path=os.path.join(st.session_state.temp_dir, "chroma")
        )
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        
        # Create vectorstore
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="rag_collection",
            embedding_function=embeddings
        )
        
        # Add documents to vectorstore
        vectorstore.add_documents(doc_splits)
        
        # Initialize components
        llm, retrieval_grade_prompt, hallucination_prompt, answer_prompt, rewrite_prompt = initialize_components()
        
        # Setup RAG chain
        rag_prompt = hub.pull("rlm/rag-prompt")
        rag_chain = rag_prompt | llm | StrOutputParser()
        
        return vectorstore, rag_chain, llm, retrieval_grade_prompt, hallucination_prompt, answer_prompt, rewrite_prompt
        
    except Exception as e:
        st.error(f"Error in setup_rag_pipeline: {str(e)}")
        raise e

def create_graph(vectorstore, rag_chain, llm, retrieval_grade_prompt, hallucination_prompt, answer_prompt, rewrite_prompt):
    # Node functions
    def retrieve(state):
        question = state["question"]
        documents = vectorstore.as_retriever().get_relevant_documents(question)
        return {"documents": documents, "question": question}
    
    def generate(state):
        question = state["question"]
        documents = state["documents"]
        docs_content = "\n\n".join([doc.page_content for doc in documents])
        generation = rag_chain.invoke({"context": docs_content, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(state):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        for doc in documents:
            score = retrieval_grade_prompt | llm | StrOutputParser()
            grade = score.invoke({"question": question, "document": doc.page_content})
            if "yes" in grade.lower():
                filtered_docs.append(doc)
        return {"documents": filtered_docs, "question": question}
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    
    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Streamlit UI
st.title("RAG Application with Self-Verification")

# Main content
st.subheader("Add Documents")

# URL input with comma separation
st.write("Enter URLs (separated by commas)")
urls_input = st.text_area("URLs", placeholder="https://example1.com, https://example2.com, https://example3.com", height=100)
urls = [url.strip() for url in urls_input.split(',') if url.strip()]

# File upload
st.write("Upload Files (PDF, DOCX, TXT, JSON)")
uploaded_files = st.file_uploader(
    "Choose files", 
    accept_multiple_files=True,
    type=['pdf', 'docx', 'txt', 'json']
)

# Initialize button
if st.button("Initialize RAG System"):
    if not urls and not uploaded_files:
        st.warning("Please provide at least one URL or upload a file.")
    else:
        with st.spinner("Initializing RAG system..."):
            try:
                vectorstore, rag_chain, llm, retrieval_grade_prompt, hallucination_prompt, answer_prompt, rewrite_prompt = setup_rag_pipeline(urls, uploaded_files)
                if vectorstore:
                    app = create_graph(vectorstore, rag_chain, llm, retrieval_grade_prompt, hallucination_prompt, answer_prompt, rewrite_prompt)
                    st.session_state.app = app
                    st.success("RAG system initialized!")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {str(e)}")

# Question input and generation
if 'app' in st.session_state:
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")
    
    if st.button("Generate Answer") and question:
        with st.spinner("Generating answer..."):
            # Create progress bar
            progress_bar = st.progress(0)
            
            try:
                # Process the question
                outputs = []
                for i, output in enumerate(st.session_state.app.stream({"question": question})):
                    outputs.append(output)
                    # Update progress
                
                # Display final answer
                if outputs:
                    final_state = outputs[-1]
                    st.subheader("Answer:")
                    st.write(final_state[list(final_state.keys())[0]]["generation"])
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
