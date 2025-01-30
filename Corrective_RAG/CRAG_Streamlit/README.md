# 📄 Corrective-RAG: Question Answering System with Streamlit

---

## ✨ Features

- Supports document retrieval from URLs and uploaded files (PDF, DOCX, TXT, JSON).
- Uses ChromaDB as a vector store for efficient document searching.
- Employs HuggingFace sentence embeddings for semantic similarity.
- Integrates Groq API for intelligent summarization and corrective query handling.
- Streamlit-based front-end for a user-friendly interface.
- CORS-enabled for cross-origin requests.

---

## 📦 Installation

Follow these steps to set up the project locally:

```sh
git clone https://github.com/adityadeshpande03/RAG/tree/main/Corrective_RAG/CRAG_Streamlit
cd CRAG_Streamlit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

To run the Streamlit app, execute the following command:

```sh
streamlit run main.py
```

This will start the Streamlit server on your local machine, typically accessible via `http://localhost:8501`.

---

## 📌 Usage

### Streamlit Interface

Once the application is running, you'll see the following options in the Streamlit interface:

- **Question Input**: Enter your question in the provided text input field.
- **File Upload**: Upload files (PDF, DOCX, TXT, JSON) for additional context.
- **URL Input**: Optionally, enter comma-separated URLs for additional document retrieval.
- **API Configuration**: Input your Groq and Tavily API keys in the sidebar.
- **Enable Web Search**: Toggle the checkbox to enable or disable web search for real-time context retrieval.

The app will use the uploaded files and/or URLs to fetch relevant documents and provide an AI-generated response with reasoning.

---

## 🛠️ Technologies Used

- **Streamlit**: User-friendly interface for building data apps.
- **ChromaDB**: Vector store for efficient document retrieval.
- **HuggingFace Sentence Transformers**: Generates semantic embeddings for text.
- **Groq**: Powers summarization and intelligent corrective query responses.
- **PyPDFLoader, Docx2txtLoader, JSONLoader**: File parsing tools for document processing.
- **Tavily Search**: Web search tool for real-time context retrieval.

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

Made with ❤️ by Adi | [GitHub Repository](https://github.com/adityadeshpande03/RAG/tree/main/Corrective_RAG/CRAG_Streamlit)
