"""
Модели базы данных и работа с SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import json

# Создание базы данных (SQLite)
DATABASE_URL = "sqlite:///./transactions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Модель пользователя"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Transaction(Base):
    """Модель транзакции"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=True)
    predicted_category = Column(String, nullable=False)
    user_category = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)
    processing_time_ms = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_auto = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    def get_final_category(self) -> str:
        return self.user_category if self.user_category else self.predicted_category
    
    def to_dict(self) -> dict:
        """Преобразование в словарь с локальным временем"""
    # Преобразуем UTC в локальное время (Москва UTC+3)
        local_time = self.created_at + timedelta(hours=3)
        return {
            'id': self.id,
            'description': self.description,
            'amount': self.amount,
            'category': self.get_final_category(),
            'predicted_category': self.predicted_category,
            'user_category': self.user_category,
            'confidence': self.confidence,
            'date': local_time.strftime('%Y-%m-%d %H:%M:%S'),
            'notes': self.notes
    }


class TrainingFeedback(Base):
    """Модель для хранения обратной связи для дообучения"""
    __tablename__ = "training_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=True)
    correct_category = Column(String, nullable=False)
    wrong_category = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    used_for_training = Column(Boolean, default=False)


# Настройка связей после определения классов
User.transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
User.feedbacks = relationship("TrainingFeedback", back_populates="user", cascade="all, delete-orphan")
Transaction.user = relationship("User", back_populates="transactions")
TrainingFeedback.user = relationship("User", back_populates="feedbacks")


def create_tables():
    """Создание всех таблиц в базе данных"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Генератор сессии базы данных"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TransactionRepository:
    """Репозиторий для работы с транзакциями"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, user_id: int, description: str, predicted_category: str, 
               confidence: float, amount: float = None, processing_time_ms: float = 0, 
               is_auto: bool = True) -> Transaction:
        """Создание новой транзакции"""
        transaction = Transaction(
            user_id=user_id,
            description=description,
            amount=amount,
            predicted_category=predicted_category,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            is_auto=is_auto
        )
        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(transaction)
        return transaction
    
    def get_user_transactions(self, user_id: int, limit: int = 100, offset: int = 0) -> List[Transaction]:
        """Получение транзакций пользователя"""
        return self.db.query(Transaction).filter(
            Transaction.user_id == user_id
        ).order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()
    
    def get_user_transactions_by_source(self, user_id: int, limit: int = 100, offset: int = 0, is_auto:
    bool = True) -> List[Transaction]:
        """Получение транзакций пользователя по источнику (ручной ввод или выписка)"""
        return self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.is_auto == is_auto
        ).order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()
    
    def update_category(self, transaction_id: int, user_id: int, new_category: str) -> Optional[Transaction]:
        """Обновление категории транзакции (ручная коррекция)"""
        transaction = self.db.query(Transaction).filter(
            Transaction.id == transaction_id,
            Transaction.user_id == user_id
        ).first()
        
        if transaction:
            transaction.user_category = new_category
            self.db.commit()
            self.db.refresh(transaction)
            
            feedback = TrainingFeedback(
                user_id=user_id,
                description=transaction.description,
                amount=transaction.amount,
                correct_category=new_category,
                wrong_category=transaction.predicted_category
            )
            self.db.add(feedback)
            self.db.commit()
            
        return transaction
    
    def delete_transaction(self, transaction_id: int, user_id: int) -> bool:
        """Удаление транзакции"""
        transaction = self.db.query(Transaction).filter(
            Transaction.id == transaction_id,
            Transaction.user_id == user_id
        ).first()
        
        if transaction:
            self.db.delete(transaction)
            self.db.commit()
            return True
        return False
    
    def get_statistics(self, user_id: int) -> dict:
        """Получение статистики по ВСЕМ транзакциям"""
        return self.get_statistics_by_source(user_id, is_auto=None)
    
    def get_statistics_by_source(self, user_id: int, is_auto: bool = None) -> dict:
        """Получение статистики по источнику (ручной ввод или выписка)"""
        query = self.db.query(Transaction).filter(Transaction.user_id == user_id)
        
        if is_auto is not None:
            query = query.filter(Transaction.is_auto == is_auto)
        
        transactions = query.all()
        
        total_expenses = sum(abs(t.amount) for t in transactions if t.amount and t.amount < 0)
        total_income = sum(t.amount for t in transactions if t.amount and t.amount > 0)
        
        category_stats = {}
        for t in transactions:
            category = t.get_final_category()
            amount = abs(t.amount) if t.amount and t.amount < 0 else 0
            if category not in category_stats:
                category_stats[category] = {'count': 0, 'total': 0}
            category_stats[category]['count'] += 1
            category_stats[category]['total'] += amount
        
        return {
            'total_transactions': len(transactions),
            'total_expenses': total_expenses,
            'total_income': total_income,
            'avg_confidence': sum(t.confidence for t in transactions) / len(transactions) if transactions else 0,
            'category_stats': category_stats,
            'corrected_count': sum(1 for t in transactions if t.user_category is not None)
        }
    
    def delete_all_transactions(self, user_id: int) -> int:
        """Удаление всех транзакций пользователя"""
        transactions = self.db.query(Transaction).filter(Transaction.user_id == user_id).all()
        count = len(transactions)
        
        for t in transactions:
            self.db.delete(t)
        
        self.db.commit()
        return count