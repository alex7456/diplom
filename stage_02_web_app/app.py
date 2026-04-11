"""
FastAPI веб-приложение с БД (упрощённая версия без авторизации для теста)
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
import time
import pandas as pd
from io import StringIO

from database import create_tables, get_db, TransactionRepository
from ml_service import EnhancedExpenseClassifier
from analytics import AnalyticsService

# Создание приложения
app = FastAPI(
    title="Expense Categorizer API",
    description="Автоматическая категоризация банковских транзакций",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Шаблоны
templates = Jinja2Templates(directory="templates")

# Инициализация ML сервиса
ml_service = EnhancedExpenseClassifier()

# Создание таблиц БД
create_tables()


# ============= Загрузка модели =============

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске"""
    import os
    print("Загрузка ML модели...")
    
    if os.path.exists('enhanced_model.pkl'):
        ml_service.load_model('enhanced_model.pkl')
        print("Модель загружена из файла")
    else:
        metrics = ml_service.train(use_synthetic=True)
        ml_service.save_model('enhanced_model.pkl')
        print(f"Модель обучена с нуля. Accuracy: {metrics['accuracy_mean']:.3f}")


# ============= Веб-страницы =============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/app", response_class=HTMLResponse)
async def app_main(request: Request):
    """Главное приложение"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "categories": ml_service.categories
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Дашборд аналитики"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request
    })


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    """История транзакций"""
    return templates.TemplateResponse("history.html", {
        "request": request
    })


# ============= API эндпоинты (без авторизации) =============

@app.post("/api/predict")
async def predict_transaction(
    description: str = Form(...),
    amount: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """Предсказание и сохранение транзакции"""
    start_time = time.time()
    
    category, confidence = ml_service.predict(description, amount)
    processing_time = (time.time() - start_time) * 1000
    
    # Сохраняем в БД (user_id = 1 для всех, упрощённо)
    repo = TransactionRepository(db)
    transaction = repo.create(
        user_id=1,
        description=description,
        amount=amount,
        predicted_category=category,
        confidence=confidence,
        processing_time_ms=processing_time
    )
    
    return {
        "id": transaction.id,
        "description": description,
        "amount": amount,
        "predicted_category": category,
        "confidence": confidence,
        "processing_time_ms": processing_time
    }


@app.get("/api/transactions")
async def get_transactions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Получение истории транзакций"""
    repo = TransactionRepository(db)
    transactions = repo.get_user_transactions(1, limit, offset)
    return [t.to_dict() for t in transactions]


@app.put("/api/transactions/{transaction_id}/category")
async def update_transaction_category(
    transaction_id: int,
    category: str,
    db: Session = Depends(get_db)
):
    """Ручная коррекция категории"""
    repo = TransactionRepository(db)
    transaction = repo.update_category(transaction_id, 1, category)
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {"success": True, "transaction": transaction.to_dict()}


@app.delete("/api/transactions/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    db: Session = Depends(get_db)
):
    """Удаление транзакции"""
    repo = TransactionRepository(db)
    success = repo.delete_transaction(transaction_id, 1)
    
    if not success:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {"success": True}


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Статистика пользователя"""
    repo = TransactionRepository(db)
    return repo.get_statistics(1)


@app.get("/api/analytics/categories")
async def get_category_stats(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Статистика по категориям"""
    analytics = AnalyticsService(db)
    return analytics.get_expenses_by_category(1, days)


@app.get("/api/analytics/monthly")
async def get_monthly_trend(
    months: int = 6,
    db: Session = Depends(get_db)
):
    """Динамика по месяцам"""
    analytics = AnalyticsService(db)
    return analytics.get_monthly_trend(1, months)


@app.get("/api/analytics/top-merchants")
async def get_top_merchants(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Топ магазинов"""
    analytics = AnalyticsService(db)
    merchants = analytics.get_top_merchants(1, limit)
    return [{"name": m[0], "total": m[1], "count": m[2]} for m in merchants]


@app.get("/api/analytics/breakdown")
async def get_breakdown(db: Session = Depends(get_db)):
    """Полная разбивка расходов"""
    analytics = AnalyticsService(db)
    return analytics.get_category_breakdown(1)


@app.get("/api/analytics/daily")
async def get_daily_spending(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Ежедневные расходы"""
    analytics = AnalyticsService(db)
    return analytics.get_daily_spending(1, days)


@app.get("/api/export/csv")
async def export_csv(days: int = 90, db: Session = Depends(get_db)):
    """Экспорт транзакций в CSV"""
    analytics = AnalyticsService(db)
    df = analytics.get_export_data(1, days)
    
    from fastapi.responses import StreamingResponse
    stream = StringIO()
    df.to_csv(stream, index=False, encoding='utf-8-sig')
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=transactions.csv"
    return response


@app.get("/api/export/excel")
async def export_excel(days: int = 90, db: Session = Depends(get_db)):
    """Экспорт транзакций в Excel"""
    analytics = AnalyticsService(db)
    df = analytics.get_export_data(1, days)
    
    from fastapi.responses import StreamingResponse
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Транзакции', index=False)
    output.seek(0)
    
    response = StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response.headers["Content-Disposition"] = "attachment; filename=transactions.xlsx"
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)