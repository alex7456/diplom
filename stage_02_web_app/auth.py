"""
JWT авторизация с поддержкой кук
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import secrets

from database import User, get_db

# Конфигурация
SECRET_KEY = "your-secret-key-change-in-production-2024-diplom-project"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 дней

security = HTTPBearer(auto_error=False)


def get_password_hash(password: str) -> str:
    """Хеширование пароля"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((password + salt).encode())
    return f"{salt}:{hash_obj.hexdigest()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля"""
    try:
        salt, hash_value = hashed_password.split(":")
        new_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
        return new_hash == hash_value
    except:
        return False


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Аутентификация пользователя"""
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создание JWT токена"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """
    Получение текущего пользователя (сначала из заголовка, потом из куки)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    
    token = None
    
    # Пробуем получить токен из заголовка Authorization
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
    
    # Если нет в заголовке, пробуем из куки
    if not token:
        token = request.cookies.get("access_token")
    
    if not token:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Получение активного пользователя"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def create_user(db: Session, email: str, username: str, password: str) -> User:
    """Создание нового пользователя"""
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    
    hashed_password = get_password_hash(password)
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user