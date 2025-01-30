RAG-Based Question Answering API

This is a FastAPI-based RAG (Retrieval-Augmented Generation) application that allows users to ask questions based on retrieved data from URLs and uploaded files. The system uses Groq LLM and ChromaDB for document retrieval and embedding storage.

Features

Accepts user questions via API

Supports document retrieval from URLs and uploaded files (PDF, DOCX, TXT, JSON)

Uses ChromaDB as a vector store

Employs HuggingFace sentence embeddings

Uses Groq's LLM for answer generation

Installation

Prerequisites

Ensure you have Python 3.8+ installed.

Setup

# Clone the repository
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

Configuration

Set up your environment variables:

export GROQ_API_KEY='your_groq_api_key'

Running the Application

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

API Documentation

POST /ask_question

Description: Allows users to ask a question based on documents from URLs or uploaded files.

Request Body

Parameter

Type

Required

Description

question

string

Yes

The question to be asked

urls

list

No

List of URLs to retrieve content from

files

UploadFile[]

No

List of files to be uploaded

Example Request

curl -X 'POST' \
  'http://localhost:8000/ask_question' \
  -H 'Content-Type: multipart/form-data' \
  -F 'question="What is climate change?"' \
  -F 'urls=https://example.com/article' \
  -F 'files=@sample.pdf'

Example Response

{
  "answer": "Climate change refers to long-term changes in temperature, precipitation, and other atmospheric conditions on Earth."
}

Contributing

Feel free to submit pull requests or report issues.

License

MIT License. See LICENSE file for details.

