"""
Authentication and authorization utilities for First Aid Assistant API
"""
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from pymongo.collection import Collection

from utils.logger_config import setup_logger

# Suppress bcrypt version warning
warnings.filterwarnings("ignore", message=".*trapped.*error reading bcrypt version.*")

logger = setup_logger(__name__)

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    secret_key: str,
    algorithm: str,
    expires_delta: Optional[timedelta] = None,
    default_expire_minutes: int = 30
) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token
        secret_key: Secret key for encoding
        algorithm: JWT algorithm
        expires_delta: Custom expiration time
        default_expire_minutes: Default expiration in minutes
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=default_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    
    return encoded_jwt


def create_refresh_token(
    data: dict,
    secret_key: str,
    algorithm: str,
    expire_days: int = 7
) -> str:
    """
    Create JWT refresh token
    
    Args:
        data: Data to encode in token
        secret_key: Secret key for encoding
        algorithm: JWT algorithm
        expire_days: Expiration in days
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=expire_days)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


async def get_current_user_optional(
    token: Optional[str],
    users_collection: Collection,
    secret_key: str,
    algorithm: str
):
    """
    Get current user if token exists, otherwise return None for guest users
    
    Args:
        token: JWT token
        users_collection: MongoDB users collection
        secret_key: Secret key for decoding
        algorithm: JWT algorithm
        
    Returns:
        User document or None
    """
    if token is None:
        return None
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError as e:
        logger.debug(f"Token validation failed: {str(e)}")
        return None
    
    user = users_collection.find_one({"username": username})
    return user


async def get_current_user(
    token: str,
    users_collection: Collection,
    secret_key: str,
    algorithm: str
):
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if token is None:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        raise credentials_exception
    
    user = users_collection.find_one({"username": token_data.username})
    if user is None:
        logger.warning(f"User not found: {token_data.username}")
        raise credentials_exception
    
    return user