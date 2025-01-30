# 📄 Corrective-RAG: API-Based Question Answering System

---

## ✨ Features

- Supports document retrieval from URLs and uploaded files (PDF, DOCX, TXT, JSON).
- Uses ChromaDB as a vector store for efficient document searching.
- Employs HuggingFace sentence embeddings for semantic similarity.
- Integrates Groq API for intelligent summarization and corrective query handling.
- FastAPI-based backend for seamless API interaction.
- CORS-enabled for cross-origin requests.

---

## 📦 Installation

Follow these steps to set up the project locally:

```sh
git clone https://github.com/adityadeshpande03/RAG/tree/main/Corrective_RAG/CRAG_Fastapi
cd CRAG_Fastapi

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the API

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📌 Usage

### API Endpoints

#### 1️⃣ Corrective Query API
- **Endpoint:** `/api/query`
- **Method:** `POST`
- **Parameters:**
  - `question` (str): The question to ask.
  - `use_web_search` (bool, optional): Whether to use web search for additional context.
  - `urls` (str, optional): Comma-separated list of URLs to retrieve documents from.
  - `files` (UploadFile, optional): Uploaded document files (PDF, TXT, JSON, DOCX).
- **Response:**
  - `response` (str): AI-generated answer with corrective reasoning.
  - `documents` (List[str]): Context used for answering.

Example request:
```sh
curl -X POST "http://localhost:8000/api/query" \
     -F "question=What is AI?" \
     -F "use_web_search=true"
```

#### 2️⃣ Health Check API
- **Endpoint:** `/health`
- **Method:** `GET`
- **Response:** `{ "status": "healthy" }`

---

## 🛠️ Technologies Used

- **FastAPI**: High-performance backend framework.
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

Made with ❤️ by Adi | [GitHub Repository](https://github.com/adityadeshpande03/RAG/tree/main/Corrective_RAG/CRAG_Fastapi)

