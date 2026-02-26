# Advanced Centrix Support – Professional README

## 1. Project Overview

Advanced Centrix Support is an AI-powered mental health assistant platform built using Python, Flask, machine learning models, and a Hybrid Retrieval-Augmented Generation (RAG) architecture. The application combines conversational AI, predictive mental health analysis, Redis-backed caching, and reranked embedding retrieval to deliver fast, context-aware, and production-grade responses.

The system integrates backend AI processing with a web-based frontend interface to create a scalable intelligent support environment.

---

## 2. Core Features

### 2.1 Hybrid RAG Pipeline

The platform implements a production-ready Hybrid RAG architecture designed for high performance and contextual accuracy.

#### Redis DB Semantic Cache System

**Multi-Level Cache Design**

The RAG pipeline uses a two-layer caching architecture:

**L1 Cache – In-Process Memory**

* Implemented using OrderedDict.
* Stores most recent semantic query embeddings.
* Provides ultra-fast retrieval for repeated queries.
* Uses cosine similarity to detect semantically similar requests.

**L2 Cache – Redis Vector Database**

* Uses RediSearch HNSW index.
* Stores query, embedding, response, sources, and timestamp.
* Performs semantic KNN search on embeddings.
* Automatically promotes hits to L1 cache.

**Redis Index Fields**

* query
* response
* sources
* timestamp
* embedding (FLOAT32 vector)

**Key Features**

* Semantic cache lookup before retrieval stage.
* TTL-based expiration for entries.
* Vector similarity threshold control.
* Exact and semantic matching supported.

#### Ranking Embedding System (Two Modes)

The retrieval system uses dual-mode embedding and ranking:

**Mode 1 – Fast Retrieval Mode**

* Lightweight embedding search for rapid document matching.
* Used for quick conversational queries.
* Optimized for low latency.

**Mode 2 – Advanced Reranked Mode**

* Initial vector retrieval followed by reranking logic.
* Semantic similarity scoring improves answer relevance.
* Ideal for complex or twisted user queries.
* Enhances contextual accuracy using embedding relationships.

**High-Level Flow**

User Query
→ Query Classification
→ Embedding Generation
→ Redis Semantic Cache (L1 + L2)
→ Vector Retrieval
→ Cross-Encoder Reranking
→ Context Compression
→ LLM Response Generation
→ Cache Writeback

---

### 2.2 AI Chat and Mental Health Assistance

* Conversational AI powered by prompt engineering.
* Emotion-aware response generation.
* Session-based chat history management.

### 2.3 Machine Learning Prediction System

* Mental health classification using a trained ML model.
* Feature encoding handled via label encoders.
* Prediction pipeline implemented in `ml_predictor.py`.

### 2.4 Self-Care and Support Modules

* Mental exercises and wellness suggestions.
* Personalized self-care planning.
* Task manager for daily improvement tracking.

### 2.5 Content Retrieval and Knowledge Base

* Document-driven support resources.
* Context-aware knowledge injection into prompts.
* Modular retrieval pipeline inside `app/content_retrieval.py`.

### 2.6 Web Application

* Flask backend (`server.py`).
* HTML templates and static frontend assets.
* Multi-page interface including Home, Help, Resources, and Contact.

---

## 3. Project Architecture

The application follows a modular, production-oriented architecture.

**Backend Stack**

* Python
* Flask
* Redis Cache Layer
* Embedding-based Retrieval Pipeline

**Machine Learning**

* Pickle-based trained models
* Feature encoding and prediction pipeline

**Frontend**

* HTML
* CSS
* JavaScript

**Database**

* SQLite (`healing.db`)
* JSON conversation storage

**System Flow**

Frontend Interface → Flask Server → Redis Cache Layer → Hybrid RAG Retrieval → ML Predictor → LLM Prompt Engine → Response

---

## 4. Folder Structure

```
Advanced_Centrix_Support/
│
├── server.py                 # Main Flask application
├── ml_predictor.py           # ML prediction logic
├── prompt.py                 # Prompt engineering system
├── requirements.txt
│
├── app/
│   ├── content_retrieval.py  # RAG retrieval pipeline
│   ├── mental_exercises.py
│   ├── self_care_plan.py
│   └── task_manager.py
│
├── templates/
├── static/
├── conversations/
│
├── mental_health_model.pkl
├── label_encoders.pkl
├── feature_names.pkl
├── Mental Health Dataset.csv
└── healing.db
```

---

## 5. Installation and Setup

### 5.1 Create Virtual Environment

```
python -m venv venv
```

Activate environment:

```
venv\Scripts\activate
```

### 5.2 Install Dependencies

```
pip install -r requirements.txt
```

### 5.3 Redis Setup (Required for RAG Cache)

Ensure Redis server is running locally or remotely.

```
redis-server
```

Update connection settings inside the backend configuration if needed.

---

## 6. Running the Application

Start the Flask server:

```
python server.py
```

Open browser:

```
http://localhost:5000
```

---

## 7. Hybrid RAG Technical Details

### Retrieval Strategy

* Semantic embedding search retrieves relevant documents.
* Redis caching layer intercepts repeated queries.
* Reranking improves document ordering before prompt assembly.

### Performance Advantages

* Reduced latency via Redis cache hits.
* Improved semantic understanding using embedding similarity.
* Adaptive mode selection depending on query complexity.

### Mode Activation

**Fast Retrieval Mode**

* Simple or direct queries.
* Cached or short conversational requests.

**Advanced Reranked Mode**

* Multi-context queries.
* Emotion-based or analytical prompts.
* Complex sentence structures.

---

## 8. Machine Learning Components

**Model Files**

* `mental_health_model.pkl`
* `label_encoders.pkl`
* `feature_names.pkl`

**Prediction Flow**

1. User input processed.
2. Features encoded.
3. Mental health category predicted.
4. Output injected into conversational pipeline.

---

## 9. Data and Storage

* Conversations stored as JSON.
* SQLite database manages structured records.
* Training dataset included for reference.

---

## 10. Security Considerations

* Protect Redis endpoints from public exposure.
* Secure conversation logs due to sensitive mental health content.
* Avoid committing environment variables.

---

## 11. Future Enhancements

* Adaptive RAG memory layer.
* Dynamic embedding updates.
* Distributed Redis clustering.
* Docker-based deployment pipeline.
* Advanced emotion-aware reranking.

---

## 12. Contribution Guidelines

1. Maintain modular architecture.
2. Test RAG retrieval and caching layers before deployment.
3. Follow structured commit practices.
4. Document new modules and embedding pipelines.

---

## 13. License

This project is intended for educational and development purposes. Review licensing requirements before production deployment.

---
