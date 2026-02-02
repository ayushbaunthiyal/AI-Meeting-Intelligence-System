# 🎯 AI Meeting Intelligence System

A conversational AI assistant that analyzes meeting transcripts and answers questions about discussions, decisions, and action items. Features voice-to-transcript capability using local Whisper (no API costs for transcription).

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🚀 Quick Setup

### Prerequisites
- Docker Desktop installed and running
- OpenAI API key

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd AI-Meeting-Intelligence-System

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run with Docker

```bash
# Build and start all services
docker-compose up --build

# View logs
docker-compose logs -f
```

### 3. Access the Application
- **UI**: http://localhost:8504
- **API Docs**: http://localhost:8001/docs

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (Port 8501)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ File Upload │  │   Analysis  │  │  Q&A Chat   │  │  Transcript │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                       REST API Routes                           ││
│  │  /upload  /transcribe  /analyze  /ask  /meetings                ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐│
│  │ Whisper       │  │ ChromaDB      │  │ LangGraph Agent           ││
│  │ (Local V2T)   │  │ (Vector Store)│  │                           ││
│  │               │  │               │  │ ┌─────────┐ ┌───────────┐ ││
│  │ • Audio→Text  │  │ • Embeddings  │  │ │ Parser  │→│Summarizer │ ││
│  │ • Timestamps  │  │ • Semantic    │  │ └─────────┘ └───────────┘ ││
│  │ • No API cost │  │   Search      │  │      ↓           ↓        ││
│  │               │  │ • In-memory   │  │ ┌─────────┐ ┌───────────┐ ││
│  └───────────────┘  └───────────────┘  │ │Decisions│→│  Actions  │ ││
│                                        │ └─────────┘ └───────────┘ ││
│                                        └───────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        OpenAI Services                          ││
│  │  LLM: gpt-3.5-turbo  │  Embeddings: text-embedding-3-small     ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### LangGraph Agent Pipeline

The system uses a 4-node LangGraph pipeline for meeting analysis:

1. **Parser Node**: Normalizes transcript format (speaker labels, timestamps)
2. **Summarizer Node**: Generates high-level meeting overview
3. **Decision Extractor Node**: Identifies key decisions and agreements
4. **Action Item Agent Node**: Extracts tasks with owners and deadlines

---

## 📁 Project Structure

```
AI-Meeting-Intelligence-System/
├── backend/
│   ├── src/
│   │   ├── agents/           # LangGraph orchestration
│   │   │   ├── nodes/        # Parser, Summarizer, Decisions, Actions
│   │   │   ├── graph.py      # Main LangGraph workflow
│   │   │   ├── qa_agent.py   # Q&A RAG agent
│   │   │   └── state.py      # Agent state definitions
│   │   ├── api/              # FastAPI routes
│   │   ├── config/           # Configuration management
│   │   ├── models/           # Pydantic schemas
│   │   ├── services/         # Whisper, LLM, Embedding services
│   │   ├── vectorstore/      # ChromaDB integration
│   │   └── main.py           # Application entry point
│   ├── Dockerfile
│   └── pyproject.toml
├── ui/
│   ├── src/
│   │   └── app.py            # Streamlit application
│   ├── Dockerfile
│   └── pyproject.toml
├── data/
│   └── sample_transcripts/   # Example meeting transcripts
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🔧 RAG/LLM Approach & Decisions

### Choices Considered

| Component | Options Evaluated | Final Choice | Rationale |
|-----------|-------------------|--------------|-----------|
| **LLM** | OpenAI GPT-3.5/4, Ollama, Claude | **OpenAI GPT-3.5-turbo** | Best balance of quality, speed, and cost |
| **Embeddings** | OpenAI, Sentence Transformers, Cohere | **OpenAI text-embedding-3-small** | Optimized for OpenAI LLM, low cost |
| **Vector DB** | ChromaDB, FAISS, Pinecone, Weaviate | **ChromaDB (in-memory)** | Simple, free, great for prototyping |
| **Voice-to-Text** | Whisper API, Whisper Local, AssemblyAI | **Whisper Local (base)** | 100% free, no API costs, good accuracy |
| **Orchestration** | LangChain LCEL, LangGraph, Custom | **LangGraph** | Stateful, debuggable, perfect for multi-step |

### Prompt & Context Management

- **Chunking**: 1000 tokens with 200 overlap for optimal retrieval
- **Context Window**: Top 5 relevant chunks for Q&A
- **System Prompts**: Role-specific prompts for each agent node
- **Structured Output**: JSON-based extraction for decisions/actions

### Guardrails

- Input validation via Pydantic models
- Maximum transcript length limits (15K tokens per call)
- Graceful error handling with fallbacks
- Source attribution for all Q&A responses

### Observability

- Structured logging with Python logging
- Health check endpoints for monitoring
- Processing time tracking for analysis
- Error tracking in agent state

---

## 🛠️ Key Technical Decisions

### 1. Separate UI and Backend
- **Why**: Clean separation of concerns, independent scaling
- **Benefit**: Can swap UI framework without touching backend

### 2. Singleton Services Pattern
- **Why**: Expensive model loading (Whisper, embeddings)
- **Benefit**: Load once, reuse across requests

### 3. In-Memory Vector Store
- **Why**: Simplicity for prototype, no infrastructure needed
- **Trade-off**: Data lost on restart (acceptable for demo)

### 4. LangGraph over LCEL
- **Why**: Complex multi-step analysis needs state management
- **Benefit**: Easy to debug, add nodes, modify flow

### 5. Whisper Base Model
- **Why**: Balance between accuracy and speed
- **Trade-off**: "base" is 74MB, "small" is 244MB - base is faster

---

## 🏭 Productionization Guide

### For AWS Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Architecture                         │
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐ │
│  │   Route53   │────▶│  CloudFront │────▶│  Application Load   │ │
│  │    (DNS)    │     │    (CDN)    │     │     Balancer        │ │
│  └─────────────┘     └─────────────┘     └─────────────────────┘ │
│                                                   │               │
│                      ┌────────────────────────────┼───────────┐  │
│                      │           ECS Cluster      │           │  │
│                      │  ┌─────────────┐  ┌───────▼─────────┐  │  │
│                      │  │ UI Service  │  │ Backend Service │  │  │
│                      │  │ (Fargate)   │  │   (Fargate)     │  │  │
│                      │  └─────────────┘  └─────────────────┘  │  │
│                      └────────────────────────────────────────┘  │
│                                                   │               │
│  ┌─────────────┐     ┌─────────────┐     ┌───────▼─────────────┐ │
│  │ Secrets Mgr │     │  ElastiCache│     │  OpenSearch or     │ │
│  │ (API Keys)  │     │  (Redis)    │     │  Pinecone (Vectors)│ │
│  └─────────────┘     └─────────────┘     └─────────────────────┘ │
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐ │
│  │    S3       │     │ CloudWatch  │     │     RDS/Aurora     │ │
│  │ (Files)     │     │ (Logging)   │     │   (Metadata)       │ │
│  └─────────────┘     └─────────────┘     └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Changes for Production

| Component | Current | Production |
|-----------|---------|------------|
| Vector Store | ChromaDB in-memory | Pinecone/Weaviate/OpenSearch |
| File Storage | Local filesystem | S3 |
| Metadata | In-memory dict | PostgreSQL/Aurora |
| Caching | None | Redis/ElastiCache |
| Secrets | .env file | Secrets Manager |
| Logging | Console | CloudWatch/Datadog |
| GPU | CPU | g4dn instances for Whisper |

### Scaling Considerations

1. **Horizontal Scaling**: Add more ECS tasks behind ALB
2. **GPU for Whisper**: Use g4dn.xlarge for faster transcription
3. **Async Processing**: Add SQS for long-running analysis
4. **CDN**: CloudFront for static assets

---

## 📋 Engineering Standards

- ✅ **Type Hints**: Full Python type annotations throughout
- ✅ **Pydantic Models**: All data structures validated
- ✅ **Dependency Injection**: Via constructor injection
- ✅ **Configuration**: Pydantic Settings with environment validation
- ✅ **Logging**: Structured logging with context
- ✅ **Error Handling**: Custom exceptions with proper handling
- ✅ **Code Style**: Black + Ruff for formatting
- ✅ **Documentation**: Docstrings for all public functions
- ✅ **Docker**: Multi-stage builds, health checks

---

## 🔮 What I'd Do Differently With More Time

1. **Speaker Diarization**: Use pyannote-audio to identify who's speaking
2. **Real-time Transcription**: WebSocket streaming for live meetings
3. **Persistent Storage**: PostgreSQL for metadata, proper vector DB
4. **Authentication**: OAuth2 with session management
5. **Caching Layer**: Redis for LLM response caching
6. **Quality Guardrails**: Answer relevance scoring, hallucination detection
7. **Meeting Templates**: Pre-defined formats (standup, planning, 1:1)
8. **Integrations**: Zoom, Teams, Google Meet imports
9. **Export**: PDF/DOCX generation for summaries
10. **Testing**: Comprehensive unit and integration tests

---

## 📖 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/transcripts/upload` | Upload text transcript |
| POST | `/api/v1/transcripts/upload-file` | Upload transcript file |
| POST | `/api/v1/audio/transcribe` | Transcribe audio (Whisper) |
| POST | `/api/v1/meetings/{id}/analyze` | Run full analysis |
| POST | `/api/v1/meetings/{id}/ask` | Ask question (Q&A) |
| GET | `/api/v1/meetings` | List all meetings |
| GET | `/api/v1/meetings/{id}` | Get meeting details |
| DELETE | `/api/v1/meetings/{id}` | Delete meeting |
| GET | `/api/v1/health` | Health check |

---

## 🧪 Testing

### Sample Transcripts

The `data/sample_transcripts/` folder contains example transcripts:
- `product_roadmap_meeting.txt` - Product planning discussion
- `engineering_standup.txt` - Engineering team standup

### Local Development (without Docker)

```bash
# Backend
cd backend
uv venv
uv pip install -e ".[dev]"
uv run uvicorn src.main:app --reload

# UI (in another terminal)
cd ui
uv venv
uv pip install -e ".[dev]"
uv run streamlit run src/app.py
```

---

## 📄 License

MIT License - See LICENSE file for details.
