# 📄 RAPTOR-RAG: API-Based Question Answering System

---

## ✨ Features

- Supports document retrieval from URLs with recursive crawling.
- Uses ChromaDB as a vector store for efficient document searching.
- Employs HuggingFace sentence embeddings for semantic similarity.
- Integrates Groq API for intelligent summarization and RAG-based query handling.
- Uses UMAP and Gaussian Mixture Model (GMM) for hierarchical text clustering.
- FastAPI-based backend for seamless API interaction.
- CORS-enabled for cross-origin requests.

---

## 📦 Installation

Follow these steps to set up the project locally:

```sh
git clone https://github.com/adityadeshpande03/RAG/new/main/RAPTOR_RAG/RAPTOR_RAG_URLS
cd RAPTOR_RAG_URLS

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the API

```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📌 Usage

### API Endpoints

#### 1️⃣ URL Processing API
- **Endpoint:** `/process/`
- **Method:** `POST`
- **Parameters:**
  - `urls` (List[str]): List of URLs to fetch documents from.
  - `max_depth` (int, optional): Depth level for recursive crawling (default: 20).
  - `question` (str): The question to ask based on retrieved documents.
- **Response:**
  - `answer` (str): AI-generated answer using RAG.
  - `source_count` (int): Number of documents processed.

Example request:
```sh
curl -X POST "http://localhost:8000/process/" \
     -H "Content-Type: application/json" \
     -d '{"urls": ["https://example.com"], "max_depth": 10, "question": "What is AI?"}'
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
- **Groq**: Powers summarization and intelligent query responses.
- **UMAP & Gaussian Mixture Model (GMM)**: Used for hierarchical text clustering.
- **BeautifulSoup**: Parses HTML content for text extraction.
- **LangChain**: Framework for building AI-powered applications.

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

Made with ❤️ by Adi | [GitHub Repository](https://github.com/adityadeshpande03/RAG/new/main/RAPTOR_RAG/RAPTOR_RAG_URLS)
