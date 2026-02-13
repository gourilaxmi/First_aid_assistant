import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient, DESCENDING

from RAG.rag import FirstAidRAGAssistant
from api.auth import (
    oauth2_scheme,
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    get_current_user_optional,
    get_current_user
)
from api.conversation import (
    create_conversation,
    update_conversation,
    save_chat_messages
)

from utils.logger_config import setup_logger, get_default_log_file

load_dotenv()

logger = setup_logger(
    __name__,
    log_level=logging.INFO,
    log_file=get_default_log_file('api')
)

# =============================================================================
# Configuration
# =============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
MAX_CONVERSATIONS_PER_USER = 10


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    email: str
    username: str
    full_name: Optional[str] = None
    created_at: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: UserResponse


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 10
    min_score: float = 0.60


class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[dict]
    confidence_score: float
    chunks_found: int
    avg_relevance: float
    performance: dict
    conversation_id: Optional[str] = None
    timestamp: str


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    user_id: str
    message_count: int
    last_query: Optional[str] = None
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    role: str
    content: str
    sources: Optional[List[dict]] = None
    confidence_score: Optional[float] = None
    timestamp: str


class UpdateTitleRequest(BaseModel):
    title: str


app = FastAPI(
    title="First Aid Assistant API",
    description="AI-powered first aid assistant with RAG capabilities",
    version="3.0.0"
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:4173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Database Setup
# =============================================================================

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    logger.critical("MONGODB_URI environment variable is required")
    raise ValueError("MONGODB_URI environment variable is required")

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['first_aid_db']
users_collection = db['users']
conversations_collection = db['conversations']
chat_history_collection = db['chat_history']

users_collection.create_index("username", unique=True)
users_collection.create_index("email", unique=True)
conversations_collection.create_index([("user_id", 1), ("created_at", DESCENDING)])
conversations_collection.create_index("conversation_id", unique=True)
chat_history_collection.create_index([("conversation_id", 1), ("timestamp", 1)])

logger.info("Database indexes created successfully")

# Initialize RAG Assistant

logger.info("Initializing First Aid RAG Assistant")

try:
    rag_assistant = FirstAidRAGAssistant()
    logger.info("RAG Assistant initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Assistant: {e}", exc_info=True)
    rag_assistant = None

# FIX: Proper async dependency functions 

async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)):
    """Dependency for optional authentication (guest + logged in users)"""
    return await get_current_user_optional(
        token, users_collection, SECRET_KEY, ALGORITHM
    )


async def get_required_user(token: str = Depends(oauth2_scheme)):
    """Dependency for required authentication (logged in users only)"""
    return await get_current_user(
        token, users_collection, SECRET_KEY, ALGORITHM
    )

# Authentication Endpoints

@app.post("/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    if users_collection.find_one({"email": user.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user.password)

    user_doc = {
        "user_id": user_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat()
    }

    users_collection.insert_one(user_doc)
    logger.info(f"New user registered: {user.username}")

    access_token = create_access_token(
        data={"sub": user.username},
        secret_key=SECRET_KEY,
        algorithm=ALGORITHM
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username},
        secret_key=SECRET_KEY,
        algorithm=ALGORITHM
    )

    user_response = UserResponse(
        user_id=user_id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        created_at=user_doc["created_at"]
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=user_response
    )


@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"username": form_data.username})

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"User logged in: {form_data.username}")

    access_token = create_access_token(
        data={"sub": user["username"]},
        secret_key=SECRET_KEY,
        algorithm=ALGORITHM
    )
    refresh_token = create_refresh_token(
        data={"sub": user["username"]},
        secret_key=SECRET_KEY,
        algorithm=ALGORITHM
    )

    user_response = UserResponse(
        user_id=user["user_id"],
        email=user["email"],
        username=user["username"],
        full_name=user.get("full_name"),
        created_at=user["created_at"]
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=user_response
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_required_user)):
    return UserResponse(
        user_id=current_user["user_id"],
        email=current_user["email"],
        username=current_user["username"],
        full_name=current_user.get("full_name"),
        created_at=current_user["created_at"]
    )


@app.post("/auth/logout")
async def logout():
    return {"message": "Logged out successfully"}


# Query Endpoint

@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: Optional[dict] = Depends(get_optional_user)  # FIX: proper async dep
):
    if not rag_assistant:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant service unavailable"
        )

    start_time = time.time()

    conversation_id = request.conversation_id
    if current_user and not conversation_id:
        conversation_id = f"conv_{current_user['user_id']}_{int(datetime.utcnow().timestamp())}"

    user_label = f"user {current_user['username']}" if current_user else "guest"
    logger.info(f"Processing query from {user_label}: {request.query[:50]}...")

    try:
        result = rag_assistant.answer_query(
            query=request.query,
            conversation_id=conversation_id,
            top_k=request.top_k,
            min_score=request.min_score,
            verbose=False
        )

        # Parse confidence score
        confidence = result.get('confidence', 'Unknown')
        if isinstance(confidence, str) and '%' in confidence:
            confidence_percentage = float(confidence.replace('%', ''))
        else:
            confidence_percentage = float(confidence) if confidence != 'Unknown' else 0.0

        # Save to DB if authenticated
        if current_user:
            conversation_exists = conversations_collection.find_one({
                "conversation_id": conversation_id
            })

            if not conversation_exists:
                create_conversation(
                    conversation_id,
                    current_user["user_id"],
                    request.query,
                    conversations_collection,
                    chat_history_collection,
                    MAX_CONVERSATIONS_PER_USER
                )
            else:
                update_conversation(
                    conversation_id,
                    request.query,
                    conversations_collection
                )

            save_chat_messages(
                conversation_id,
                request.query,
                result['response'],
                result.get('sources', []),
                confidence_percentage,
                chat_history_collection
            )

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Response generated in {total_time:.0f}ms")

        return QueryResponse(
            query=result.get('query', request.query),
            response=result['response'],
            sources=result.get('sources', []),
            confidence_score=confidence_percentage,
            chunks_found=result.get('chunks_found', 0),
            avg_relevance=result.get('avg_relevance', 0.0),
            performance=result.get('performance', {}),
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/api/conversations")
async def get_conversations(
    limit: int = 20,
    current_user: dict = Depends(get_required_user)  # FIX
):
    conversations = list(
        conversations_collection.find({"user_id": current_user["user_id"]})
        .sort("updated_at", -1)
        .limit(min(limit, MAX_CONVERSATIONS_PER_USER))
    )

    for conv in conversations:
        conv.pop("_id", None)

    return {"conversations": conversations}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_required_user)  # FIX
):
    conv = conversations_collection.find_one({
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"]
    })

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = list(
        chat_history_collection.find({"conversation_id": conversation_id})
        .sort("timestamp", 1)
    )

    for msg in messages:
        msg.pop("_id", None)

    conv.pop("_id", None)

    return {"conversation": conv, "messages": messages}


@app.get("/api/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    current_user: dict = Depends(get_required_user)  # FIX
):
    conv = conversations_collection.find_one({
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"]
    })

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = list(
        chat_history_collection.find({"conversation_id": conversation_id})
        .sort("timestamp", 1)
        .limit(limit)
    )

    for msg in messages:
        msg.pop("_id", None)

    return {
        "conversation_id": conversation_id,
        "message_count": len(messages),
        "messages": messages
    }


@app.put("/api/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    request: UpdateTitleRequest,
    current_user: dict = Depends(get_required_user)  # FIX
):
    result = conversations_collection.update_one(
        {
            "conversation_id": conversation_id,
            "user_id": current_user["user_id"]
        },
        {
            "$set": {
                "title": request.title,
                "updated_at": datetime.utcnow().isoformat()
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Title updated successfully"}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_required_user)  # FIX
):
    result = conversations_collection.delete_one({
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"]
    })

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    chat_history_collection.delete_many({"conversation_id": conversation_id})
    logger.info(f"Deleted conversation {conversation_id}")

    return {"message": "Conversation deleted successfully"}


@app.get("/")
async def root():
    return {
        "name": "First Aid Assistant API",
        "version": "3.0.0",
        "status": "running",
        "rag_available": rag_assistant is not None,
        "max_conversations_per_user": MAX_CONVERSATIONS_PER_USER
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if rag_assistant else "degraded",
        "rag_available": rag_assistant is not None,
        "database": "connected",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    logger.info("Starting First Aid Assistant API")
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info(f"Docs: http://localhost:{args.port}/docs")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )