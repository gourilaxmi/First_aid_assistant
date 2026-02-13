# First Aid Assistant

An AI-powered first aid guidance platform built with a RAG (Retrieval-Augmented Generation) pipeline, FastAPI backend, and React frontend. Provides evidence-based first aid information sourced from authoritative medical references.


---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)

---

## Features

- **RAG-powered responses** using BioBERT embeddings + Pinecone vector search
- **Conversation history** — save, rename, and delete past consultations
- **User authentication** — register, login, JWT-based sessions
- **Guest mode** — use without an account (no history saved)
- **Confidence scoring** — each response includes a confidence level
- **Medical sources** — data from Red Cross, WHO, NHS, Mayo Clinic, and more

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Vite, Tailwind CSS, Axios |
| Backend | FastAPI, Python, Uvicorn |
| AI/ML | BioBERT, Groq (Llama 3.3 70B), LangChain |
| Vector DB | Pinecone |
| Database | MongoDB |
| Auth | JWT (python-jose), bcrypt |

---

## Project Structure

```
first_aid_assistant/
├── backend/
│   ├── RAG/
│   │   ├── embeddings.py        # BioBERT embedding generation
│   │   ├── query_processor.py   # Query processing pipeline
│   │   ├── rag.py               # Core RAG assistant
│   │   └── response_generator.py # LLM response generation
│   ├── api/
│   │   ├── auth.py              # JWT authentication utilities
│   │   └── conversation.py      # Conversation management
│   ├── collectors/              # Data collection pipeline
│   ├── utils/
│   │   └── logger_config.py     # Logging configuration
│   ├── main.py                  # FastAPI application
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   └── ConversationHistory.jsx
│   │   ├── contexts/
│   │   │   └── AuthContext.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── LoginPage.jsx
│   │   └── RegisterPage.jsx
│   ├── package.json
│   └── .gitignore
├── tests/
└── .gitignore
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Conda (recommended) or virtualenv
- MongoDB Atlas account
- Pinecone account
- Groq API key

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/gourilaxmi/First_aid_assistant.git
cd First_aid_assistant
```

### 2. Backend setup

```bash
# Create and activate conda environment
conda create -n first_aid python=3.11
conda activate first_aid

# Install dependencies
cd backend
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
```

### 3. Frontend setup

```bash
cd frontend
npm install

# Copy and fill in environment variables
cp frontend.env.example .env
```

---

## Environment Variables

### Backend (`backend/.env`)

```env
# MongoDB
MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Groq
GROQ_API_KEY=your_groq_api_key

# JWT
SECRET_KEY=your_secret_key_change_in_production

# CORS (comma-separated)
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Frontend (`frontend/.env`)

```env
VITE_API_URL=http://localhost:8000
```

---

## Running the Application

### Backend

```bash
cd backend
conda activate first_aid
uvicorn main:app --reload
```

Backend runs at: `http://localhost:8000`  
API docs at: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login (returns JWT) |
| GET | `/auth/me` | Get current user info |
| POST | `/auth/logout` | Logout |

### Query
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Submit a first aid query |

### Conversations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List all conversations |
| GET | `/api/conversations/{id}` | Get conversation with messages |
| PUT | `/api/conversations/{id}/title` | Update conversation title |
| DELETE | `/api/conversations/{id}` | Delete a conversation |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |

---

## Notes

- The `first_aid/` virtual environment folder is excluded from git — always use your own conda/venv
- Log files in `backend/logs/` are excluded from git
- Never commit `.env` files — use `.env.example` as a template
