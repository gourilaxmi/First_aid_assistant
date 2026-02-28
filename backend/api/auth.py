import logging
import bcrypt
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from pymongo.collection import Collection

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password[:72].encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    password_bytes = password[:72].encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')


def create_access_token(
    data: dict,
    secret_key: str,
    algorithm: str,
    expires_delta: Optional[timedelta] = None,
    default_expire_minutes: int = 30
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=default_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret_key, algorithm=algorithm)


def create_refresh_token(
    data: dict,
    secret_key: str,
    algorithm: str,
    expire_days: int = 7
) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=expire_days)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret_key, algorithm=algorithm)


async def get_current_user_optional(
    token: Optional[str],
    users_collection: Collection,
    secret_key: str,
    algorithm: str
):
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
    return users_collection.find_one({"username": username})


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