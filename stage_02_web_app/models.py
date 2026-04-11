"""
Pydantic модели для API
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class CategoryEnum(str, Enum):
    """Доступные категории"""
    PRODUCTS = "Продукты"
    TRANSPORT = "Транспорт"
    CAFE = "Кафе"
    TAXI = "Такси"
    PHARMACY = "Аптеки"
    COMMUNICATION = "Связь"
    UTILITIES = "Коммунальные"
    ENTERTAINMENT = "Развлечения"
    CLOTHING = "Одежда"
    TRANSFERS = "Переводы"
    OTHER = "Другое"


class TransactionRequest(BaseModel):
    """Запрос на предсказание одной транзакции"""
    description: str = Field(..., description="Описание транзакции", min_length=1, max_length=500)
    amount: Optional[float] = Field(None, description="Сумма транзакции (положительная для дохода, отрицательная для расхода)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Пятерочка 1234",
                "amount": -450.50
            }
        }


class TransactionResponse(BaseModel):
    """Ответ с предсказанием"""
    description: str
    amount: Optional[float]
    predicted_category: str
    confidence: float
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchRequest(BaseModel):
    """Пакетный запрос"""
    transactions: List[TransactionRequest]


class BatchResponse(BaseModel):
    """Пакетный ответ"""
    predictions: List[TransactionResponse]
    total_time_ms: float
    average_confidence: float


class TrainingDataRequest(BaseModel):
    """Запрос на дообучение"""
    description: str
    correct_category: str
    amount: Optional[float] = None


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса"""
    status: str
    model_loaded: bool
    model_accuracy: Optional[float]
    version: str = "2.0.0"