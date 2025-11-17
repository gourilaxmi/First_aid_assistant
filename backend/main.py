from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uvicorn
import time
import uuid
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Local imports
from RAG.rag import FirstAidRAGAssistant
from metrics.performance_tracker import tracker

load_dotenv()

# ============================================================================
# Security Configuration
# ============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="First Aid Assistant API",
    description="AI-powered first aid assistant with RAG",
    version="3.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Database Setup
# ============================================================================

MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['first_aid_db']
users_collection = db['users']
conversations_collection = db['conversations']
chat_history_collection = db['chat_history']

# Create indexes
users_collection.create_index("username", unique=True)
users_collection.create_index("email", unique=True)
conversations_collection.create_index("user_id")
try:
    conversations_collection.create_index("conversation_id", unique=True)
except Exception as e:
    print(f"⚠️ Skipping index creation: {e}")
chat_history_collection.create_index("conversation_id")

# ============================================================================
# Initialize RAG Assistant
# ============================================================================

print("\nInitializing First Aid RAG Assistant...")
try:
    rag_assistant = FirstAidRAGAssistant()
    print("✓ RAG Assistant ready!\n")
except Exception as e:
    print(f"Failed to initialize RAG Assistant: {e}")
    rag_assistant = None

# ============================================================================
# Pydantic Models
# ============================================================================

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

class TokenData(BaseModel):
    username: Optional[str] = None

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

# ============================================================================
# Auth Helper Functions
# ============================================================================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=180)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)):
    """Get current user if token exists, otherwise return None for guest users"""
    if token is None:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    
    user = users_collection.find_one({"username": username})
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user (required authentication)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if token is None:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = users_collection.find_one({"username": token_data.username})
    if user is None:
        raise credentials_exception
    return user

# ============================================================================
# Auth Endpoints
# ============================================================================

@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate):
    """Register a new user"""
    
    # Check if user exists
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    hashed_password = get_password_hash(user.password)
    
    user_doc = {
        "user_id": user_id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "is_active": True
    }
    
    users_collection.insert_one(user_doc)
    
    return UserResponse(
        user_id=user_id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        created_at=user_doc["created_at"]
    )

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user"""
    
    # Find user by username or email
    user = users_collection.find_one({
        "$or": [
            {"username": form_data.username},
            {"email": form_data.username}
        ]
    })
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user["username"]})
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            username=user["username"],
            full_name=user.get("full_name"),
            created_at=user["created_at"]
        )
    )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return UserResponse(
        user_id=current_user["user_id"],
        email=current_user["email"],
        username=current_user["username"],
        full_name=current_user.get("full_name"),
        created_at=current_user["created_at"]
    )

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user (client should remove tokens)"""
    return {"message": "Successfully logged out"}

# ============================================================================
# Query Endpoints
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Process query - works for both authenticated users and guests"""
    
    if not rag_assistant:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Assistant not available"
        )
    
    start_time = time.time()
    
    try:
        # Generate conversation ID if not provided and user is authenticated
        conversation_id = request.conversation_id
        if not conversation_id and current_user:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        
        # Process query with RAG
        result = rag_assistant.answer_query(
            query=request.query,
            conversation_id=conversation_id,
            top_k=request.top_k,
            verbose=False
        )
        
        # Use current timestamp for accurate timing
        current_timestamp = datetime.utcnow().isoformat()
        
        # Calculate confidence as percentage (0-100)
        avg_score = result.get('avg_relevance', 0.0)
        confidence_percentage = round(avg_score * 100, 1)
        
        # Save to conversation history if user is logged in
        if current_user and conversation_id:
            # Check if conversation exists
            conv = conversations_collection.find_one({
                "conversation_id": conversation_id
            })
            
            if not conv:
                # Create new conversation
                conversations_collection.insert_one({
                    "conversation_id": conversation_id,
                    "user_id": current_user["user_id"],
                    "title": request.query[:60] + "..." if len(request.query) > 60 else request.query,
                    "message_count": 2,
                    "last_query": request.query,
                    "created_at": current_timestamp,
                    "updated_at": current_timestamp
                })
            else:
                # Update existing conversation
                conversations_collection.update_one(
                    {"conversation_id": conversation_id},
                    {
                        "$set": {
                            "last_query": request.query,
                            "updated_at": current_timestamp
                        },
                        "$inc": {"message_count": 2}
                    }
                )
            
            # Save messages to chat history
            chat_history_collection.insert_many([
                {
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": request.query,
                    "timestamp": current_timestamp
                },
                {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": result['response'],
                    "sources": result['sources'],
                    "confidence_score": confidence_percentage,
                    "timestamp": current_timestamp
                }
            ])
        
        response = QueryResponse(
            query=result['query'],
            response=result['response'],
            sources=result['sources'],
            confidence_score=confidence_percentage,
            chunks_found=result['chunks_found'],
            avg_relevance=result['avg_relevance'],
            performance=result.get('performance', {}),
            conversation_id=conversation_id,
            timestamp=current_timestamp
        )
        
        # Track performance
        total_time = (time.time() - start_time) * 1000
        tracker.record_query(request.query, result, True, total_time)
        
        return response
        
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        tracker.record_query(request.query, {}, False, total_time)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

# ============================================================================
# Conversation Management Endpoints
# ============================================================================

@app.get("/api/conversations")
async def get_conversations(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get all conversations for current user"""
    
    conversations = list(conversations_collection.find(
        {"user_id": current_user["user_id"]}
    ).sort("updated_at", -1).limit(limit))
    
    for conv in conversations:
        conv.pop("_id", None)
    
    return {"conversations": conversations}

@app.get("/api/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get chat history for a conversation"""
    
    # Verify conversation belongs to user
    conv = conversations_collection.find_one({
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"]
    })
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get chat history
    messages = list(chat_history_collection.find(
        {"conversation_id": conversation_id}
    ).sort("timestamp", 1).limit(limit))
    
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
    title: str,
    current_user: dict = Depends(get_current_user)
):
    """Update conversation title"""
    
    result = conversations_collection.update_one(
        {
            "conversation_id": conversation_id,
            "user_id": current_user["user_id"]
        },
        {
            "$set": {
                "title": title,
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
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation"""
    
    # Delete conversation
    result = conversations_collection.delete_one({
        "conversation_id": conversation_id,
        "user_id": current_user["user_id"]
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete chat history
    chat_history_collection.delete_many({"conversation_id": conversation_id})
    
    return {"message": "Conversation deleted successfully"}

# ============================================================================
# Performance Metrics Endpoints
# ============================================================================

@app.get("/api/metrics")
async def get_metrics():
    """Get performance metrics"""
    stats = tracker.get_stats()
    return {
        "components": {k: v for k, v in stats.items() if k != 'overall'},
        "overall": stats.get('overall', {}),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/metrics/reset")
async def reset_metrics():
    """Reset metrics"""
    tracker.reset()
    return {"message": "Metrics reset successfully"}

# ============================================================================
# Health & Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "First Aid Assistant API",
        "version": "3.0.0",
        "status": "running",
        "rag_available": rag_assistant is not None
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if rag_assistant else "degraded",
        "rag_available": rag_assistant is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚑 FIRST AID ASSISTANT ")
    print("="*70)
    print(f"\n🌐 Server: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://localhost:{args.port}/docs")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )