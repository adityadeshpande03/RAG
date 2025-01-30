# 📄 Self-RAG : Based Question Answering API with Streamlit UI

---

## ✨ Features

- Supports document retrieval from URLs and uploaded files (PDF, DOCX, TXT, JSON).
- Uses ChromaDB as a vector store for efficient document searching.
- Employs HuggingFace sentence embeddings for semantic similarity.
- Integrates Groq API for intelligent summarization and query handling.
- Interactive Streamlit UI for seamless user interaction.

---

## 📦 Installation

Follow these steps to set up the project locally:

```sh
git clone https://github.com/adityadeshpande03/RAG/tree/main/Self_RAG/Self_RAG_Streamlit
cd Self_RAG_Streamlit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```sh
streamlit run app.py
```

---

## 📌 Usage

1. Launch the app by running `streamlit run app.py`.
2. Enter URLs or upload documents in supported formats.
3. Ask a question in the chat interface.
4. Get instant answers powered by Groq and Sentence Transformers.

---

## 🛠️ Technologies Used

- **Streamlit**: Interactive UI for document-based Q&A.
- **ChromaDB**: Vector store for efficient document retrieval.
- **HuggingFace Sentence Transformers**: Generates semantic embeddings for text.
- **Groq**: Powers summarization and intelligent query responses.
- **PyPDFLoader, Docx2txtLoader, JSONLoader**: File parsing tools for document processing.

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

Made with ❤️ by Adi | [GitHub Repository](https://github.com/adityadeshpande03/RAG/tree/main/Self_RAG/Self_RAG_Streamlit)

