# DocuQuery AI — Multi-PDF RAG Chatbot

> **Author:** Smeet Patel | M.Tech CSE, Delhi Technological University | 2026

An end-to-end GenAI application that lets users upload multiple PDF documents and ask natural-language questions over them. The system uses Retrieval-Augmented Generation (RAG) with LangChain, FAISS vector search, and Google Gemini to generate context-aware answers grounded in uploaded documents.

## Core Features
- Multi-PDF upload and parsing
- Chunking and vector indexing with FAISS
- Semantic search over document content
- Context-aware answers with Gemini
- Streamlit web interface for interactive Q&A
- Chat history and source-aware response flow

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Google Gemini
- Sentence Transformers
- PyPDF

## Project Structure
```
pdf-rag-chatbot/
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   └── utils.py
├── data/
├── vectorstore/
├── requirements.txt
├── .python-version
├── .gitignore
├── Procfile
└── README.md
```

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
```

Run locally:
```bash
streamlit run app/main.py
```

## Resume Positioning
This project demonstrates practical GenAI engineering by combining document ingestion, semantic retrieval, prompt orchestration, vector search, and an interactive frontend in a single deployable application.
