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

from database import TrainingFeedback, Transaction, create_tables, get_db, TransactionRepository
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
    """Загрузка модели при запуске с авто-дообучением"""
    import os
    print("Загрузка ML модели...")
    
    if os.path.exists('enhanced_model.pkl'):
        ml_service.load_model('enhanced_model.pkl')
        print("Модель загружена из файла")
        
        # Проверяем наличие неиспользованных исправлений
        from database import SessionLocal
        db = SessionLocal()
        try:
            auto_retrain_if_needed(db)
        finally:
            db.close()
    else:
        metrics = ml_service.train(use_synthetic=True)
        ml_service.save_model('enhanced_model.pkl')
        print(f"Модель обучена с нуля. Точность: {metrics.get('accuracy', 0):.3f}")


# ============= Веб-страницы =============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница (лендинг)"""
    return templates.TemplateResponse("landing.html", {"request": request})


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
    date: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Предсказание и сохранение транзакции (ручной ввод)"""
    start_time = time.time()
    
    category, confidence = ml_service.predict(description, amount)
    processing_time = (time.time() - start_time) * 1000
    
    # Обработка даты
    transaction_date = datetime.now()
    if date:
        try:
            transaction_date = datetime.strptime(date, '%Y-%m-%d')
        except:
            pass
    
    transaction = Transaction(
        user_id=1,
        description=description,
        amount=amount,
        predicted_category=category,
        confidence=confidence,
        processing_time_ms=processing_time,
        is_auto=True,
        created_at=transaction_date
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    
    return {
        "id": transaction.id,
        "description": description,
        "amount": amount,
        "predicted_category": category,
        "confidence": confidence,
        "processing_time_ms": processing_time,
        "date": transaction_date.strftime('%Y-%m-%d')
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
    from database import UploadedFile
    
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
    date_col = None
    
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
    amount_keywords = ['сумма операции', 'сумма', 'amount', 'списано', 'зачислено']
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
    
    # Ищем колонку с датой
    date_keywords = ['дата операции', 'дата', 'date', 'datetime']
    for col in df.columns:
        if any(keyword in col for keyword in date_keywords):
            date_col = col
            break
    
    # Обработка транзакций
    processed = 0
    total_expenses = 0
    total_income = 0
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
            
            # Получаем сумму (НЕ МЕНЯЕМ ЗНАК)
            amount = None
            if amount_col and pd.notna(row[amount_col]):
                try:
                    amount = float(row[amount_col])
                except:
                    amount = None
            
            # Получаем дату из выписки
            transaction_date = datetime.now()
            if date_col and pd.notna(row[date_col]):
                try:
                    transaction_date = pd.to_datetime(row[date_col])
                except:
                    pass
            
            # Предсказываем категорию
            category, confidence = ml_service.predict(description, amount)
            
            # Сохраняем в БД
            transaction = Transaction(
                user_id=1,
                description=description[:500],
                amount=amount,
                predicted_category=category,
                confidence=confidence,
                processing_time_ms=0,
                is_auto=False,
                created_at=transaction_date
            )
            db.add(transaction)
            processed += 1
            
            if amount:
                if amount < 0:
                    total_expenses += abs(amount)
                elif amount > 0:
                    total_income += amount
            
            category_stats[category] += 1
            
        except Exception as e:
            errors.append(f"Строка {idx}: {str(e)[:100]}")
            continue
    
    db.commit()
    
    if processed == 0:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Не удалось обработать ни одной транзакции. Ошибки: {errors[:3]}"}
        )
    
    # Сохраняем информацию о загруженном файле
    uploaded_file = UploadedFile(
        user_id=1,
        filename=file.filename,
        upload_date=datetime.now(),  # Текущая дата и время загрузки
        transactions_count=processed,
        total_amount=total_expenses
    )
    db.add(uploaded_file)
    db.commit()
    
    return {
        "success": True,
        "total": processed,
        "total_expenses": total_expenses,
        "total_income": total_income,
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
    """Получение истории транзакций (только ручной ввод)"""
    repo = TransactionRepository(db)
    # Фильтруем только ручные транзакции (is_auto=True)
    transactions = repo.get_user_transactions_by_source(1, limit, offset, is_auto=True)
    return [t.to_dict() for t in transactions]


@app.put("/api/transactions/{transaction_id}/category")
async def update_transaction_category(
    transaction_id: int,
    category: str,
    db: Session = Depends(get_db)
):
    """Ручная коррекция категории с мгновенным дообучением"""
    repo = TransactionRepository(db)
    transaction = repo.update_category(transaction_id, 1, category)
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Запускаем дообучение в отдельном потоке с НОВОЙ сессией
    import threading
    # Передаём копию данных, а не сессию
    threading.Thread(target=auto_retrain_if_needed, args=(db,)).start()
    
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
    """Статистика только для ручного ввода (is_auto=True)"""
    repo = TransactionRepository(db)
    return repo.get_statistics_by_source(1, is_auto=True)


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

@app.get("/api/analytics/manual")
async def get_manual_analytics(db: Session = Depends(get_db)):
    """Аналитика только для ручного ввода (is_auto = True)"""
    analytics = AnalyticsService(db)
    return analytics.get_analytics_by_source(1, is_auto=True)

@app.get("/api/analytics/bank")
async def get_bank_analytics(db: Session = Depends(get_db)):
    """Аналитика только для выписок (is_auto = False)"""
    analytics = AnalyticsService(db)
    return analytics.get_analytics_by_source(1, is_auto=False)


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

@app.post("/api/retrain")
async def retrain_model_manual(db: Session = Depends(get_db)):
    """Ручной запуск дообучения"""
    success = auto_retrain_if_needed(db)
    
    if success:
        return {
            "success": True,
            "message": "Модель успешно дообучена"
        }
    else:
        pending = db.query(TrainingFeedback).filter(
            TrainingFeedback.used_for_training == False
        ).count()
        return {
            "success": False,
            "message": f"Недостаточно исправлений. Нужно минимум 5, есть {pending}",
            "pending_count": pending
        }
    

@app.get("/api/statements")
async def get_statements_list(db: Session = Depends(get_db)):
    """Список загруженных выписок"""
    from database import UploadedFile
    
    files = db.query(UploadedFile).filter(
        UploadedFile.user_id == 1
    ).order_by(UploadedFile.upload_date.desc()).all()
    
    statements = []
    for f in files:
        statements.append({
            'filename': f.filename,
            'upload_date': f.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transactions_count': f.transactions_count,
            'total_amount': f.total_amount
        })
    
    return {"statements": statements}


@app.get("/statements", response_class=HTMLResponse)
async def statements_page(request: Request):
    """Страница управления выписками"""
    return templates.TemplateResponse("statements.html", {"request": request})

@app.get("/api/analytics/manual/weekly")
async def get_manual_weekly_spending(db: Session = Depends(get_db)):
    """Расходы по дням недели (ручной ввод)"""
    analytics = AnalyticsService(db)
    return analytics.get_weekly_spending(1, is_auto=True)

@app.get("/api/analytics/manual/predict")
async def get_manual_predict(db: Session = Depends(get_db)):
    """Прогноз расходов (ручной ввод)"""
    analytics = AnalyticsService(db)
    return analytics.predict_next_month_expenses(1, is_auto=True)

@app.get("/api/analytics/bank/predict")
async def get_bank_predict(db: Session = Depends(get_db)):
    """Прогноз расходов (выписки)"""
    analytics = AnalyticsService(db)
    return analytics.predict_next_month_expenses(1, is_auto=False)

@app.get("/api/analytics/insights")
async def get_insights(
    db: Session = Depends(get_db),
    source: str = "manual"
):
    """
    Анализ расходов и персональные рекомендации
    """
    analytics = AnalyticsService(db)
    
    # Определяем источник данных
    is_auto = True if source == "manual" else False
    
    # Проверяем, есть ли данные для выписок
    if source == "bank":
        bank_stats = analytics.get_analytics_by_source(1, is_auto=False)
        if bank_stats.get('total_transactions', 0) == 0:
            return {
                "insights": [],
                "total_expenses": 0,
                "recommendations_count": 0,
                "period_days": 30,
                "message": "Нет загруженных выписок"
            }
    
    # Получаем данные с учётом источника
    breakdown = analytics.get_category_breakdown(1, is_auto=is_auto)
    monthly = analytics.get_monthly_trend(1, 3, is_auto=is_auto)
    top_merchants = analytics.get_top_merchants(1, 10, is_auto=is_auto)
    weekly = analytics.get_weekly_spending(1, is_auto=is_auto)
    
    insights = []
    
    # 1. Анализ по категориям
    categories = breakdown.get('categories', {})
    total_expenses = breakdown.get('total_expenses', 0)
    
    norms = {
        'Кафе': 5000, 'Такси': 3000, 'Каршеринг': 2000,
        'Развлечения': 4000, 'Одежда': 8000, 'Аптеки': 2000,
        'Продукты': 15000, 'Подписки': 1000, 'Маркетплейсы': 5000
    }
    
    for category, data in categories.items():
        spent = data.get('total', 0)
        if category in norms and spent > norms[category]:
            excess = spent - norms[category]
            percent = (excess / norms[category]) * 100
            insights.append({
                "type": "warning",
                "title": f"Перерасход в категории «{category}»",
                "message": f"Вы потратили {spent:.0f} ₽, что на {excess:.0f} ₽ ({percent:.0f}%) больше нормы",
                "suggestion": f"Рекомендуем сократить расходы на {int(excess * 0.3)}-{int(excess * 0.5)} ₽"
            })
    
    # 2. Анализ частых покупок
    for merchant, total, count in top_merchants[:3]:
        if total > 3000:
            insights.append({
                "type": "info",
                "title": f"Частые покупки в «{merchant}»",
                "message": f"За последнее время вы потратили {total:.0f} ₽ ({count} покупок)",
                "suggestion": "Попробуйте поискать альтернативы или покупать по акциям"
            })
    
    # 3. Анализ кафе
    cafe_spent = categories.get('Кафе', {}).get('total', 0)
    if cafe_spent > 3000:
        savings = cafe_spent * 0.4
        insights.append({
            "type": "saving",
            "title": "Экономия на обедах",
            "message": f"Вы тратите {cafe_spent:.0f} ₽ на кафе и рестораны",
            "suggestion": f"Готовьте обед дома — можно сэкономить до {savings:.0f} ₽!"
        })
    
    # 4. Анализ транспорта
    taxi_spent = categories.get('Такси', {}).get('total', 0)
    carsharing_spent = categories.get('Каршеринг', {}).get('total', 0)
    transport_spent = taxi_spent + carsharing_spent
    
    if transport_spent > 2000:
        savings = transport_spent * 0.5
        insights.append({
            "type": "saving",
            "title": "Экономия на транспорте",
            "message": f"Вы тратите {transport_spent:.0f} ₽ на такси и каршеринг",
            "suggestion": f"Используйте общественный транспорт — экономия до {savings:.0f} ₽!"
        })
    
    # 5. Анализ подписок
    subscriptions_spent = categories.get('Подписки', {}).get('total', 0)
    if subscriptions_spent > 500:
        insights.append({
            "type": "info",
            "title": "Проверьте подписки",
            "message": f"Вы тратите {subscriptions_spent:.0f} ₽ на подписки в месяц",
            "suggestion": "Откажитесь от неиспользуемых подписок — экономия до 30%"
        })
    
    # 6. Общая рекомендация
    if total_expenses > 30000:
        savings_potential = total_expenses * 0.15
        insights.append({
            "type": "warning",
            "title": "Анализ общей картины",
            "message": f"Ваши расходы за месяц: {total_expenses:.0f} ₽",
            "suggestion": f"Потенциал экономии: ~{savings_potential:.0f} ₽"
        })
    elif 0 < total_expenses < 15000:
        insights.append({
            "type": "success",
            "title": "Отличная финансовая дисциплина!",
            "message": f"Ваши расходы ({total_expenses:.0f} ₽) ниже среднего уровня",
            "suggestion": "Продолжайте в том же духе!"
        })
    
    # 7. Основная статья расходов
    if categories:
        top_category = max(categories.items(), key=lambda x: x[1]['total'])
        if top_category[1]['total'] > total_expenses * 0.4:
            insights.append({
                "type": "info",
                "title": f"Основная статья расходов — {top_category[0]}",
                "message": f"На {top_category[0]} уходит {(top_category[1]['total']/total_expenses*100):.0f}% всех трат",
                "suggestion": "Проанализируйте, можно ли оптимизировать расходы в этой категории"
            })
    
    return {
        "insights": insights,
        "total_expenses": total_expenses,
        "recommendations_count": len(insights),
        "period_days": 30
    }


def auto_retrain_if_needed(db: Session):
    """Автоматическое дообучение после КАЖДОГО исправления"""
    from database import SessionLocal
    
    # Получаем последнее неиспользованное исправление (используем НОВУЮ сессию)
    new_db = SessionLocal()
    try:
        pending_feedback = new_db.query(TrainingFeedback).filter(
            TrainingFeedback.used_for_training == False
        ).order_by(TrainingFeedback.created_at.desc()).first()
        
        if pending_feedback is None:
            return False
        
        print(f"\n🔄 Обнаружено новое исправление: '{pending_feedback.description}' → {pending_feedback.correct_category}")
        
        # Получаем ВСЕ неиспользованные исправления
        all_pending = new_db.query(TrainingFeedback).filter(
            TrainingFeedback.used_for_training == False
        ).all()
        
        # Собираем данные для обучения
        train_data = []
        
        # Добавляем исправления с высоким весом (10 копий)
        for f in all_pending:
            for _ in range(10):
                train_data.append({
                    'description': f.description,
                    'category': f.correct_category,
                    'amount': f.amount
                })
        
        # Добавляем историю исправлений
        old_feedbacks = new_db.query(TrainingFeedback).filter(
            TrainingFeedback.used_for_training == True
        ).order_by(TrainingFeedback.created_at.desc()).limit(200).all()
        
        for f in old_feedbacks:
            for _ in range(3):
                train_data.append({
                    'description': f.description,
                    'category': f.correct_category,
                    'amount': f.amount
                })
        
        # Добавляем минимум синтетических данных
        print("   Добавление базовых синтетических данных...")
        synthetic_df = ml_service._generate_enhanced_data(100)
        for _, row in synthetic_df.iterrows():
            train_data.append({
                'description': row['description'],
                'category': row['category'],
                'amount': row['amount']
            })
        
        print(f"   Всего примеров для обучения: {len(train_data)}")
        
        # Дообучаем модель
        df = pd.DataFrame(train_data)
        ml_service.train(df=df, use_synthetic=False)
        ml_service.save_model('enhanced_model.pkl')
        
        # Очищаем кэш модели
        ml_service.cache = {}
        print("   Кэш модели очищен")
        
        # Отмечаем все неиспользованные исправления как использованные
        for f in all_pending:
            feedback = new_db.query(TrainingFeedback).filter(TrainingFeedback.id == f.id).first()
            if feedback:
                feedback.used_for_training = True
        new_db.commit()
        
        print(f"✅ Модель дообучена! Теперь '{pending_feedback.description}' → '{pending_feedback.correct_category}'")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка дообучения: {e}")
        new_db.rollback()
        return False
    finally:
        new_db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)