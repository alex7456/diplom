"""
FastAPI веб-приложение с БД (упрощённая версия без авторизации для теста)
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
import time
import pandas as pd
from io import StringIO, BytesIO
from collections import defaultdict

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


@app.post("/api/upload-bank-statement")
async def upload_bank_statement(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Загрузка банковской выписки (CSV или Excel)
    Поддерживаются форматы: Сбер, Т-Банк, Альфа, ВТБ
    """
    import chardet
    import csv
    
    contents = await file.read()
    
    # Определяем тип файла
    if file.filename.endswith('.csv'):
        # Пробуем определить кодировку
        detected = chardet.detect(contents)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        
        # Пробуем разные разделители
        separators = [',', ';', '\t', '|']
        df = None
        
        # Сначала пробуем прочитать как обычный CSV с автоопределением
        for sep in separators:
            try:
                # Пробуем прочитать с определённым разделителем
                text = contents.decode(encoding, errors='ignore')
                from io import StringIO
                df = pd.read_csv(StringIO(text), sep=sep, encoding=encoding, 
                                on_bad_lines='skip', engine='python')
                if len(df.columns) > 1:
                    print(f"Успешно прочитан CSV с разделителем '{sep}' и кодировкой {encoding}")
                    break
            except:
                continue
        
        # Если не получилось, пробуем через стандартный csv модуль
        if df is None or len(df.columns) <= 1:
            try:
                text = contents.decode(encoding, errors='ignore')
                lines = text.split('\n')
                dialect = csv.Sniffer().sniff(lines[0])
                from io import StringIO
                df = pd.read_csv(StringIO(text), dialect=dialect, on_bad_lines='skip')
                print(f"Успешно определён диалект CSV: {dialect.delimiter}")
            except:
                pass
        
        # Если всё ещё нет, пробуем без заголовка
        if df is None or len(df.columns) <= 1:
            for sep in separators:
                try:
                    text = contents.decode(encoding, errors='ignore')
                    from io import StringIO
                    df = pd.read_csv(StringIO(text), sep=sep, header=None, on_bad_lines='skip')
                    if len(df.columns) > 1:
                        # Используем первую строку как заголовок
                        df.columns = df.iloc[0]
                        df = df[1:]
                        break
                except:
                    continue
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Не удалось прочитать CSV файл. Проверьте формат.")
    
    elif file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла. Используйте CSV или Excel.")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Файл пуст")
    
    # Приводим колонки к нижнему регистру для поиска
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # Автоопределение колонок
    description_col = None
    amount_col = None
    
    # Ищем колонку с описанием
    description_keywords = ['описание', 'назначение', 'description', 'наименование', 'comment', 'название', 'detail']
    for col in df.columns:
        if any(keyword in col for keyword in description_keywords):
            description_col = col
            break
    
    # Если не нашли, берём первую текстовую колонку
    if description_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                description_col = col
                break
    
    # Ищем колонку с суммой
    amount_keywords = ['сумма', 'amount', 'списано', 'зачислено', 'payment', 'total', 'сумма операции']
    for col in df.columns:
        if any(keyword in col for keyword in amount_keywords):
            amount_col = col
            break
    
    # Если не нашли, берём числовую колонку
    if amount_col is None:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                amount_col = col
                break
    
    # Обработка транзакций
    repo = TransactionRepository(db)
    processed = 0
    total_expenses = 0
    category_stats = defaultdict(int)
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Получаем описание
            description = ""
            if description_col and pd.notna(row[description_col]):
                description = str(row[description_col]).strip()
            
            if not description or len(description) < 2:
                continue
            
            # Пропускаем технические строки
            skip_words = ['баланс', 'остаток', 'выписка', 'начало', 'конец', 'итого', 'всего', 
                         'page', 'total', 'balance', 'остаток на', 'начальный остаток']
            if any(word in description.lower() for word in skip_words):
                continue
            
            # Получаем сумму
            amount = None
            if amount_col and pd.notna(row[amount_col]):
                try:
                    amount = float(row[amount_col])
                    
                    # Определяем знак суммы по названию колонки
                    if 'списано' in str(amount_col).lower() or 'расход' in str(amount_col).lower():
                        amount = -abs(amount)
                    elif 'зачислено' in str(amount_col).lower() or 'доход' in str(amount_col).lower():
                        amount = abs(amount)
                    # Если сумма положительная, предполагаем что это расход (по умолчанию)
                    elif amount > 0:
                        amount = -amount
                except:
                    amount = None
            
            # Предсказываем категорию
            category, confidence = ml_service.predict(description, amount)
            
            # Сохраняем в БД
            repo.create(
                user_id=1,
                description=description[:500],
                amount=amount if amount else None,
                predicted_category=category,
                confidence=confidence,
                processing_time_ms=0
            )
            processed += 1
            if amount and amount < 0:
                total_expenses += abs(amount)
            category_stats[category] += 1
            
        except Exception as e:
            errors.append(f"Строка {idx}: {str(e)[:100]}")
            continue
    
    if processed == 0:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Не удалось обработать ни одной транзакции. Ошибки: {errors[:3]}"}
        )
    
    return {
        "success": True,
        "total": processed,
        "total_expenses": total_expenses,
        "category_stats": dict(category_stats),
        "filename": file.filename,
        "errors": errors[:5] if errors else []
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

@app.get("/api/transactions/clear-all")
async def clear_all_transactions_get(db: Session = Depends(get_db)):
    """Очистка всех транзакций (GET метод)"""
    from sqlalchemy import text
    
    result = db.execute(text("DELETE FROM transactions WHERE user_id = 1"))
    db.commit()
    
    return {
        "success": True,
        "deleted_count": result.rowcount,
        "message": f"Удалено {result.rowcount} транзакций"
    }

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
    
    output = BytesIO()
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