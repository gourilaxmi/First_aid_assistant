"""
Conversation management utilities for First Aid Assistant API
"""
import logging
from datetime import datetime
from pymongo.collection import Collection

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


def manage_conversation_limit(
    user_id: str,
    conversations_collection: Collection,
    chat_history_collection: Collection,
    max_conversations: int = 10
) -> None:
    """
    Ensure user has at most max_conversations conversations.
    Delete oldest conversations if limit exceeded.
    
    Args:
        user_id: User identifier
        conversations_collection: MongoDB conversations collection
        chat_history_collection: MongoDB chat history collection
        max_conversations: Maximum number of conversations to keep
    """
    conv_count = conversations_collection.count_documents({"user_id": user_id})
    
    if conv_count >= max_conversations:
        to_delete_count = conv_count - max_conversations + 1
        
        oldest_convs = list(
            conversations_collection.find({"user_id": user_id})
            .sort("created_at", 1)
            .limit(to_delete_count)
        )
        
        for conv in oldest_convs:
            conv_id = conv['conversation_id']
            
            # Delete conversation
            conversations_collection.delete_one({"conversation_id": conv_id})
            
            # Delete associated chat history
            chat_history_collection.delete_many({"conversation_id": conv_id})
            
            logger.info(
                f"Deleted old conversation for user {user_id}: {conv_id}"
            )


def create_conversation(
    conversation_id: str,
    user_id: str,
    query: str,
    conversations_collection: Collection,
    chat_history_collection: Collection,
    max_conversations: int = 10
) -> None:
    """
    Create a new conversation
    
    Args:
        conversation_id: Unique conversation identifier
        user_id: User identifier
        query: Initial query text
        conversations_collection: MongoDB conversations collection
        chat_history_collection: MongoDB chat history collection
        max_conversations: Maximum conversations to keep per user
    """
    manage_conversation_limit(
        user_id,
        conversations_collection,
        chat_history_collection,
        max_conversations
    )
    
    current_timestamp = datetime.utcnow().isoformat()
    title = query[:60] + "..." if len(query) > 60 else query
    
    conversations_collection.insert_one({
        "conversation_id": conversation_id,
        "user_id": user_id,
        "title": title,
        "message_count": 2,
        "last_query": query,
        "created_at": current_timestamp,
        "updated_at": current_timestamp
    })
    
    logger.info(f"Created conversation {conversation_id} for user {user_id}")


def update_conversation(
    conversation_id: str,
    query: str,
    conversations_collection: Collection
) -> None:
    """
    Update an existing conversation
    
    Args:
        conversation_id: Conversation identifier
        query: Latest query text
        conversations_collection: MongoDB conversations collection
    """
    current_timestamp = datetime.utcnow().isoformat()
    
    conversations_collection.update_one(
        {"conversation_id": conversation_id},
        {
            "$set": {
                "last_query": query,
                "updated_at": current_timestamp
            },
            "$inc": {"message_count": 2}
        }
    )
    
    logger.debug(f"Updated conversation {conversation_id}")


def save_chat_messages(
    conversation_id: str,
    query: str,
    response: str,
    sources: list,
    confidence_score: float,
    chat_history_collection: Collection
) -> None:
    """
    Save user query and assistant response to chat history
    
    Args:
        conversation_id: Conversation identifier
        query: User query
        response: Assistant response
        sources: Source documents
        confidence_score: Confidence percentage
        chat_history_collection: MongoDB chat history collection
    """
    current_timestamp = datetime.utcnow().isoformat()
    
    chat_history_collection.insert_many([
        {
            "conversation_id": conversation_id,
            "role": "user",
            "content": query,
            "timestamp": current_timestamp
        },
        {
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": response,
            "sources": sources,
            "confidence_score": confidence_score,
            "timestamp": current_timestamp
        }
    ])
    
    logger.debug(f"Saved messages to conversation {conversation_id}")