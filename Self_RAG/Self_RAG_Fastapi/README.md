# 📄 Self-RAG : Based Question Answering API
---

## ✨ Features

- Supports document retrieval from URLs and uploaded files (PDF, DOCX, TXT, JSON).
- Uses ChromaDB as a vector store for efficient document searching.
- Employs HuggingFace sentence embeddings for semantic similarity.
- Integrates Groq API for intelligent summarization and query handling.
- REST API interface built with FastAPI for easy integration.

---

## 📦 Installation

Follow these steps to set up the project locally:

```sh
git clone https://github.com/adityadeshpande03/RAG/tree/main/Self_RAG/Self_RAG_Fastapi
cd Self_RAG_Fastapi

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📌 API Documentation

### `POST /ask_question`

**Description:** Allows users to ask a question based on documents from URLs or uploaded files.

#### Request Body

| Parameter  | Type           | Required | Description                           |
| ---------- | -------------- | -------- | ------------------------------------- |
| `question` | `string`       | Yes      | The question to be asked              |
| `urls`     | `list`         | No       | List of URLs to retrieve content from |
| `files`    | `UploadFile[]` | No       | List of files to be uploaded          |

#### Example Request

```sh
curl -X 'POST' \
  'http://localhost:8000/ask_question' \
  -H 'Content-Type: multipart/form-data' \
  -F 'question="What is climate change?"' \
  -F 'urls=https://example.com/article' \
  -F 'files=@sample.pdf'
```

#### Example Response

```json
{
  "answer": "Climate change refers to long-term changes in temperature, precipitation, and other atmospheric conditions on Earth."
}
```

---

## 🛠️ Technologies Used

- **FastAPI**: Web framework for API development.
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

Made with ❤️ by Adi | [GitHub Repository](https://github.com/adityadeshpande03/RAG/tree/main/Self_RAG/Self_RAG_Fastapi)

