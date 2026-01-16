# CentrixSupport AI

CentrixSupport AI is a production-grade, document-aware AI support system built with Flask and Retrieval-Augmented Generation (RAG).  
It enables users to upload documents in multiple formats and ask intelligent, context-aware questions while ensuring factual accuracy, emotional awareness, and safe AI behavior.

---

## Overview

CentrixSupport AI combines modern LLM orchestration with robust backend engineering.  
The system dynamically processes documents, builds semantic vector indexes, reranks results for precision, compresses context to reduce hallucinations, and generates grounded responses using state-of-the-art language models.

The platform also includes emotion detection, multilingual support, conversation memory, and crisis-risk handling, making it suitable for sensitive domains such as mental health and enterprise knowledge systems.

---

## Key Features

### Retrieval-Augmented Generation (RAG)
- Semantic vector search using LlamaIndex
- Cross-encoder reranking for high-precision retrieval
- Context compression to remove redundancy
- Strict document-only answering to minimize hallucinations

### Multi-Format Document Ingestion
Supported file types:
- PDF, DOCX, TXT
- Images with OCR (PNG, JPG, JPEG, BMP, TIFF)
- CSV, JSON, XML
- HTML, Markdown

### Intelligent Caching System
- L1 in-memory cache and L2 disk-based cache
- Semantic similarity-based cache hits
- Automatic cache persistence and eviction

### Emotion & Language Detection
- Emotion detection with confidence scoring
- Language detection: English, Hindi, Hinglish
- High-risk crisis keyword detection with immediate safety response

### Multi-Model LLM Strategy
- Groq LLaMA 3.3 70B for high-quality English responses
- Ollama Gemma 2B fallback for local inference
- Automatic model switching based on availability

### Conversational Memory
- Session-based conversation history
- Context-aware follow-up responses
- Persistent chat storage per session

### Secure and Scalable Backend
- Flask-based REST API
- CORS-enabled architecture
- Safe file handling and cleanup
- Modular, extensible design

---

## Tech Stack

### Backend
- Python
- Flask
- Flask-CORS

### AI and NLP
- LlamaIndex
- HuggingFace Sentence Transformers
- Groq LLM
- Ollama
- Cross-Encoder Reranking

### Document Processing
- pdfplumber
- python-docx
- pytesseract
- OpenCV
- NLTK

### Infrastructure
- Vector embeddings
- Disk-based persistence
- Environment-based configuration

---

## Project Structure
.
├── server.py # Main Flask application
├── content_retrieval.py # RAG and retrieval pipeline
├── uploads/ # Temporary uploaded files (gitignored)
├── conversations/ # Session-based chat history (gitignored)
├── rag_storage/ # Vector index storage (gitignored)
├── rag_cache/ # Semantic cache storage (gitignored)
├── templates/ # HTML templates
├── static/ # CSS and JS assets
├── prompt.py # System prompts
├── .env # Environment variables (gitignored)
└── README.md


---

## Environment Variables

Create a `.env` file in the project root:
GROQ_API_KEY=your_groq_api_key
FLASK_SECRET_KEY=your_secret_key
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECEIVER_EMAIL=receiver_email@gmail.com

---
## Installation

### Clone the Repository
git clone https://github.com/yourusername/centrixsupport-ai.git

### Install Dependencies
pip install -r requirements.txt

### Run Ollama (for Hindi and Hinglish support)
ollama serve
ollama pull gemma2:2b


### Start the Application

The application runs at: http://localhost:8000

---

## API Endpoints

| Endpoint | Method | Description |
|--------|--------|------------|
| /upload | POST | Upload one or more documents |
| /search | POST | Query documents using RAG |
| /conversation/history | POST | Retrieve session chat history |
| /conversation/clear | POST | Clear session history |
| /health | GET | System health check |

---

## Safety and Responsible AI

- This system is not a medical or clinical replacement
- Crisis detection provides immediate support resources
- No diagnosis or treatment advice is generated
- All responses are either document-grounded or explicitly marked as unavailable

---

## Use Cases

- Mental health support assistants
- Enterprise document Q&A systems
- Research and compliance platforms
- Knowledge base automation
- Multilingual AI chat applications

---

## Future Enhancements

- User authentication and roles
- Streaming responses
- Frontend drag-and-drop file uploads
- Vector database integration (FAISS, Chroma)
- Docker and Kubernetes deployment

---

## License

MIT License

---

## Author

Priyanshu Shukla
Generative AI Developer
