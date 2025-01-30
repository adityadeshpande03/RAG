import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from typing import List, Dict, Any
from chromadb import PersistentClient
import re
import os
import tempfile

# Initialize Streamlit page configuration
st.set_page_config(page_title="RAG System", layout="wide")
st.title("RAG System with Groq, LangChain, and Tavily")
st.write("Ask a question and optionally provide URLs for additional context!")

# Define the RAG prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the question. If you're unsure or the context doesn't contain relevant information, say so."),
    ("user", "Context: {context}\n\nQuestion: {question}")
])

class RAGSystem:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        # Create a temporary directory for Chroma
        self.temp_dir = tempfile.mkdtemp()
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=0
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model='llama-3.3-70b-versatile',
            temperature=0
        )
        # Set Tavily API key in environment variable
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.web_search_tool = TavilySearchResults()
        self.vectorstore = None
        
    def validate_urls(self, urls_input: str) -> List[str]:
        """Validate and return list of valid URLs."""
        if not urls_input:
            return []
            
        urls = [url.strip() for url in urls_input.split(',')]
        valid_urls = []
        
        for url in urls:
            if re.match(r'^(http|https)://', url):
                valid_urls.append(url)
            else:
                st.warning(f"Invalid URL skipped: {url}")
        return valid_urls
        
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
            st.error(f"Error loading documents: {str(e)}")
            return []
            
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
            st.error(f"Error setting up vector store: {str(e)}")
            return False
            
    def web_search(self, question: str) -> List[Document]:
        """Perform web search and return results as documents."""
        try:
            results = self.web_search_tool.invoke({"query": question})
            if results:
                return [Document(page_content=result["content"]) for result in results]
            return []
        except Exception as e:
            st.error(f"Error during web search: {str(e)}")
            return []
            
    def process_query(self, question: str, use_web_search: bool = True) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        try:
            docs = []
            
            # Get documents from vector store if available
            if self.vectorstore:
                docs.extend(self.vectorstore.as_retriever().get_relevant_documents(question))
            
            # Add web search results if enabled
            if use_web_search:
                web_docs = self.web_search(question)
                docs.extend(web_docs)
            
            if not docs:
                if use_web_search:
                    return {"error": "No relevant information found from documents or web search."}
                return {"error": "Please provide valid documents or enable web search."}
            
            # Generate response
            context = "\n".join([doc.page_content for doc in docs])
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return {
                "response": response,
                "documents": docs
            }
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

# Streamlit UI components
with st.sidebar:
    st.header("API Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    tavily_api_key = st.text_input("Enter Tavily API Key", type="password")
    use_web_search = st.checkbox("Enable Web Search", value=True)

# Main content
question = st.text_input("Enter your question")
urls_input = st.text_area("Enter custom URLs for additional context (optional, separate by commas)")

if st.button("Process"):
    if not groq_api_key or not tavily_api_key:
        st.error("Please provide both API keys in the sidebar.")
    elif not question:
        st.error("Please enter a question.")
    else:
        try:
            # Initialize RAG system with API keys
            rag_system = RAGSystem(groq_api_key, tavily_api_key)
            
            # Process URLs and load documents if provided
            valid_urls = rag_system.validate_urls(urls_input)
            documents = rag_system.load_documents(valid_urls)
            
            # Setup vector store if we have documents
            if documents:
                rag_system.setup_vectorstore(documents)
            
            # Process query
            result = rag_system.process_query(question, use_web_search)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Query processed successfully!")
                st.write("Response:", result["response"])
                
                # Display retrieved documents
                with st.expander("View Retrieved Documents"):
                    for i, doc in enumerate(result["documents"]):
                        st.write(f"Document {i+1}:")
                        st.write(doc.page_content)
                        st.write("---")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")