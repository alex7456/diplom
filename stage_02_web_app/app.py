"""
FastAPI веб-приложение с БД и сессионной авторизацией
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import time
import pandas as pd
from io import StringIO, BytesIO
from collections import defaultdict
import threading
import os
import hashlib
import secrets

from database import TrainingFeedback, Transaction, User, UploadedFile, create_tables, get_db, TransactionRepository
from ml_service import EnhancedExpenseClassifier
from analytics import AnalyticsService

# Создание приложения
app = FastAPI(
    title="SmartSpend API",
    description="Автоматическая категоризация банковских транзакций",
    version="3.0.0"
)

# Добавляем middleware для сессий (ВАЖНО: до CORS)
app.add_middleware(
    SessionMiddleware,
    secret_key="smartspend-session-secret-key-2024-change-in-production",
    session_cookie="smartspend_session",
    max_age=60*60*24*7,  # 7 дней
    same_site="lax"
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


# ============= Вспомогательные функции для работы с паролями =============

def hash_password(password: str) -> str:
    """Хеширование пароля с солью"""
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


# ============= Зависимость для получения текущего пользователя из сессии =============

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Получение текущего пользователя из сессии"""
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=403, detail="Not authenticated")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=403, detail="User not found")
    return user


# ============= Загрузка модели =============

def auto_retrain_if_needed(db: Session):
    """Автоматическое дообучение модели"""
    from database import SessionLocal
    new_db = SessionLocal()
    try:
        pending_feedback = new_db.query(TrainingFeedback).filter(
            TrainingFeedback.used_for_training == False
        ).order_by(TrainingFeedback.created_at.desc()).first()
        
        if pending_feedback is None:
            return False
        
        print(f"\n🔄 Обнаружено новое исправление: '{pending_feedback.description}' → {pending_feedback.correct_category}")
        
        all_pending = new_db.query(TrainingFeedback).filter(
            TrainingFeedback.used_for_training == False
        ).all()
        
        train_data = []
        
        for f in all_pending:
            for _ in range(10):
                train_data.append({
                    'description': f.description,
                    'category': f.correct_category,
                    'amount': f.amount
                })
        
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
        
        print("   Добавление базовых синтетических данных...")
        synthetic_df = ml_service._generate_enhanced_data(100)
        for _, row in synthetic_df.iterrows():
            train_data.append({
                'description': row['description'],
                'category': row['category'],
                'amount': row['amount']
            })
        
        print(f"   Всего примеров для обучения: {len(train_data)}")
        
        df = pd.DataFrame(train_data)
        ml_service.train(df=df, use_synthetic=False)
        ml_service.save_model('enhanced_model.pkl')
        
        ml_service.cache = {}
        print("   Кэш модели очищен")
        
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


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске"""
    print("Загрузка ML модели...")
    
    if os.path.exists('enhanced_model.pkl'):
        ml_service.load_model('enhanced_model.pkl')
        print("Модель загружена из файла")
    else:
        metrics = ml_service.train(use_synthetic=True)
        ml_service.save_model('enhanced_model.pkl')
        print(f"Модель обучена с нуля. Точность: {metrics.get('accuracy', 0):.3f}")


# ============= Веб-страницы =============

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Лендинг страница"""
    # Если уже авторизован - перенаправляем в приложение
    if request.session.get("user_id"):
        return RedirectResponse(url="/app", status_code=303)
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Страница входа и регистрации"""
    # ЕСЛИ УЖЕ АВТОРИЗОВАН - ПЕРЕНАПРАВЛЯЕМ НА /app
    if request.session.get("user_id"):
        return RedirectResponse(url="/app", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


# Добавьте эти print в функции

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    print(f"🔐 Попытка входа: username={username}")
    
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        print(f"❌ Пользователь {username} не найден в БД")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Неверный логин или пароль"
        })
    
    print(f"✅ Пользователь найден: {user.username}, id={user.id}")
    
    if not verify_password(password, user.hashed_password):
        print(f"❌ Неверный пароль для {username}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Неверный логин или пароль"
        })
    
    # Сохраняем пользователя в сессии
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    
    print(f"✅ Сессия сохранена: user_id={user.id}")
    print(f"   Session data: {dict(request.session)}")
    
    return RedirectResponse(url="/app", status_code=303)


async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Получение текущего пользователя из сессии"""
    print(f"🔍 get_current_user вызван")
    print(f"   Cookie: {request.headers.get('cookie', 'Нет cookie')}")
    
    user_id = request.session.get("user_id")
    print(f"   user_id из сессии: {user_id}")
    
    if not user_id:
        print("❌ Нет user_id в сессии")
        raise HTTPException(status_code=403, detail="Not authenticated")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        print(f"❌ Пользователь с id={user_id} не найден в БД")
        raise HTTPException(status_code=403, detail="User not found")
    
    print(f"✅ Текущий пользователь: {user.username}")
    return user


@app.post("/register")
async def register(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Обработка регистрации"""
    # Проверка существующего пользователя
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Имя пользователя уже занято"
        })
    
    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Email уже зарегистрирован"
        })
    
    if len(password) < 4:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Пароль должен быть не менее 4 символов"
        })
    
    # Создаём пользователя
    user = User(
        email=email,
        username=username,
        hashed_password=hash_password(password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Автоматически входим после регистрации
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    
    return RedirectResponse(url="/app", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    """Выход из системы - полная очистка сессии"""
    # Очищаем всю сессию
    request.session.clear()
    # Создаём редирект на лендинг
    response = RedirectResponse(url="/", status_code=303)
    # Удаляем cookie с сессией на клиенте
    response.delete_cookie("smartspend_session")
    return response


@app.get("/app", response_class=HTMLResponse)
async def app_main(request: Request, current_user: User = Depends(get_current_user)):
    """Главное приложение (требует авторизацию)"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "categories": ml_service.categories,
        "user": current_user
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_user)):
    """Дашборд аналитики"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": current_user
    })


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, current_user: User = Depends(get_current_user)):
    """История транзакций"""
    return templates.TemplateResponse("history.html", {
        "request": request,
        "user": current_user
    })


@app.get("/statements", response_class=HTMLResponse)
async def statements_page(request: Request, current_user: User = Depends(get_current_user)):
    """Страница управления выписками"""
    return templates.TemplateResponse("statements.html", {
        "request": request,
        "user": current_user
    })


# ============= API эндпоинты (с авторизацией через сессию) =============

@app.post("/api/predict")
async def predict_transaction(
    request: Request,
    description: str = Form(...),
    amount: Optional[float] = Form(None),
    date: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Предсказание и сохранение транзакции"""
    start_time = time.time()
    
    category, confidence = ml_service.predict(description, amount)
    processing_time = (time.time() - start_time) * 1000
    
    transaction_date = datetime.now()
    if date:
        try:
            transaction_date = datetime.strptime(date, '%Y-%m-%d')
        except:
            pass
    
    transaction = Transaction(
        user_id=current_user.id,
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
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Загрузка банковской выписки"""
    import chardet
    import csv
    
    contents = await file.read()
    
    if file.filename.endswith('.csv'):
        detected = chardet.detect(contents)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        
        separators = [',', ';', '\t', '|']
        df = None
        
        for sep in separators:
            try:
                text = contents.decode(encoding, errors='ignore')
                df = pd.read_csv(StringIO(text), sep=sep, encoding=encoding, 
                                on_bad_lines='skip', engine='python')
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        if df is None or len(df.columns) <= 1:
            try:
                text = contents.decode(encoding, errors='ignore')
                lines = text.split('\n')
                dialect = csv.Sniffer().sniff(lines[0])
                df = pd.read_csv(StringIO(text), dialect=dialect, on_bad_lines='skip')
            except:
                pass
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Не удалось прочитать CSV файл")
    
    elif file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Файл пуст")
    
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    description_col = None
    amount_col = None
    date_col = None
    
    for col in df.columns:
        if any(word in col for word in ['описание', 'назначение', 'description', 'наименование', 'comment', 'название']):
            description_col = col
            break
    
    if description_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                description_col = col
                break
    
    for col in df.columns:
        if any(word in col for word in ['сумма операции', 'сумма', 'amount', 'списано', 'зачислено']):
            amount_col = col
            break
    
    if amount_col is None:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                amount_col = col
                break
    
    for col in df.columns:
        if any(word in col for word in ['дата операции', 'дата', 'date', 'datetime']):
            date_col = col
            break
    
    processed = 0
    total_expenses = 0
    total_income = 0
    category_stats = defaultdict(int)
    errors = []
    
    for idx, row in df.iterrows():
        try:
            description = ""
            if description_col and pd.notna(row[description_col]):
                description = str(row[description_col]).strip()
            
            if not description or len(description) < 2:
                continue
            
            skip_words = ['баланс', 'остаток', 'выписка', 'начало', 'конец', 'итого', 'total', 'balance']
            if any(word in description.lower() for word in skip_words):
                continue
            
            amount = None
            if amount_col and pd.notna(row[amount_col]):
                try:
                    amount = float(row[amount_col])
                except:
                    amount = None
            
            transaction_date = datetime.now()
            if date_col and pd.notna(row[date_col]):
                try:
                    transaction_date = pd.to_datetime(row[date_col])
                except:
                    pass
            
            category, confidence = ml_service.predict(description, amount)
            
            transaction = Transaction(
                user_id=current_user.id,
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
    
    uploaded_file = UploadedFile(
        user_id=current_user.id,
        filename=file.filename,
        upload_date=datetime.now(),
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
    request: Request,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получение истории транзакций"""
    repo = TransactionRepository(db)
    transactions = repo.get_user_transactions_by_source(current_user.id, limit, offset, is_auto=True)
    return [t.to_dict() for t in transactions]


@app.put("/api/transactions/{transaction_id}/category")
async def update_transaction_category(
    transaction_id: int,
    category: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Ручная коррекция категории"""
    repo = TransactionRepository(db)
    transaction = repo.update_category(transaction_id, current_user.id, category)
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    threading.Thread(target=auto_retrain_if_needed, args=(db,)).start()
    
    return {"success": True, "transaction": transaction.to_dict()}


@app.delete("/api/transactions/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Удаление транзакции"""
    repo = TransactionRepository(db)
    success = repo.delete_transaction(transaction_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {"success": True}


@app.get("/api/transactions/clear-all")
async def clear_all_transactions_get(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Очистка всех транзакций пользователя"""
    from sqlalchemy import text
    
    result = db.execute(text(f"DELETE FROM transactions WHERE user_id = {current_user.id}"))
    db.commit()
    
    return {
        "success": True,
        "deleted_count": result.rowcount,
        "message": f"Удалено {result.rowcount} транзакций"
    }


@app.get("/api/stats")
async def get_stats(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Статистика транзакций"""
    repo = TransactionRepository(db)
    return repo.get_statistics_by_source(current_user.id, is_auto=True)


@app.get("/api/analytics/categories")
async def get_category_stats(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Статистика по категориям"""
    analytics = AnalyticsService(db)
    return analytics.get_expenses_by_category(current_user.id, days)


@app.get("/api/analytics/monthly")
async def get_monthly_trend(
    months: int = 6,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Динамика по месяцам"""
    analytics = AnalyticsService(db)
    return analytics.get_monthly_trend(current_user.id, months)


@app.get("/api/analytics/top-merchants")
async def get_top_merchants(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Топ магазинов"""
    analytics = AnalyticsService(db)
    merchants = analytics.get_top_merchants(current_user.id, limit)
    return [{"name": m[0], "total": m[1], "count": m[2]} for m in merchants]


@app.get("/api/analytics/breakdown")
async def get_breakdown(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Полная разбивка расходов"""
    analytics = AnalyticsService(db)
    return analytics.get_category_breakdown(current_user.id)


@app.get("/api/analytics/daily")
async def get_daily_spending(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Ежедневные расходы"""
    analytics = AnalyticsService(db)
    return analytics.get_daily_spending(current_user.id, days)


@app.get("/api/analytics/manual")
async def get_manual_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Аналитика для ручного ввода"""
    analytics = AnalyticsService(db)
    return analytics.get_analytics_by_source(current_user.id, is_auto=True)


@app.get("/api/analytics/bank")
async def get_bank_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Аналитика для выписок"""
    analytics = AnalyticsService(db)
    return analytics.get_analytics_by_source(current_user.id, is_auto=False)


@app.get("/api/analytics/manual/weekly")
async def get_manual_weekly_spending(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Расходы по дням недели"""
    analytics = AnalyticsService(db)
    return analytics.get_weekly_spending(current_user.id, is_auto=True)


@app.get("/api/analytics/manual/predict")
async def get_manual_predict(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Прогноз расходов"""
    analytics = AnalyticsService(db)
    return analytics.predict_next_month_expenses(current_user.id, is_auto=True)


@app.get("/api/analytics/bank/predict")
async def get_bank_predict(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Прогноз расходов для выписок"""
    analytics = AnalyticsService(db)
    return analytics.predict_next_month_expenses(current_user.id, is_auto=False)


@app.get("/api/analytics/insights")
async def get_insights(
    db: Session = Depends(get_db),
    source: str = "manual",
    current_user: User = Depends(get_current_user)
):
    """Анализ расходов и рекомендации"""
    analytics = AnalyticsService(db)
    
    is_auto = True if source == "manual" else False
    
    if source == "bank":
        bank_stats = analytics.get_analytics_by_source(current_user.id, is_auto=False)
        if bank_stats.get('total_transactions', 0) == 0:
            return {
                "insights": [],
                "total_expenses": 0,
                "recommendations_count": 0,
                "period_days": 30,
                "message": "Нет загруженных выписок"
            }
    
    breakdown = analytics.get_category_breakdown(current_user.id, is_auto=is_auto)
    total_expenses = breakdown.get('total_expenses', 0)
    categories = breakdown.get('categories', {})
    top_merchants = analytics.get_top_merchants(current_user.id, 10, is_auto=is_auto)
    
    insights = []
    
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
    
    for merchant, total, count in top_merchants[:3]:
        if total > 3000:
            insights.append({
                "type": "info",
                "title": f"Частые покупки в «{merchant}»",
                "message": f"За последнее время вы потратили {total:.0f} ₽ ({count} покупок)",
                "suggestion": "Попробуйте поискать альтернативы или покупать по акциям"
            })
    
    cafe_spent = categories.get('Кафе', {}).get('total', 0)
    if cafe_spent > 3000:
        savings = cafe_spent * 0.4
        insights.append({
            "type": "saving",
            "title": "Экономия на обедах",
            "message": f"Вы тратите {cafe_spent:.0f} ₽ на кафе и рестораны",
            "suggestion": f"Готовьте обед дома — можно сэкономить до {savings:.0f} ₽!"
        })
    
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
    
    subscriptions_spent = categories.get('Подписки', {}).get('total', 0)
    if subscriptions_spent > 500:
        insights.append({
            "type": "info",
            "title": "Проверьте подписки",
            "message": f"Вы тратите {subscriptions_spent:.0f} ₽ на подписки в месяц",
            "suggestion": "Откажитесь от неиспользуемых подписок — экономия до 30%"
        })
    
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


@app.get("/api/export/csv")
async def export_csv(
    days: int = 90,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Экспорт транзакций в CSV"""
    analytics = AnalyticsService(db)
    df = analytics.get_export_data(current_user.id, days)
    
    stream = StringIO()
    df.to_csv(stream, index=False, encoding='utf-8-sig')
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=transactions.csv"
    return response


@app.get("/api/export/excel")
async def export_excel(
    days: int = 90,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Экспорт транзакций в Excel"""
    analytics = AnalyticsService(db)
    df = analytics.get_export_data(current_user.id, days)
    
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


@app.get("/api/statements")
async def get_statements_list(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Список загруженных выписок"""
    files = db.query(UploadedFile).filter(
        UploadedFile.user_id == current_user.id
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


@app.post("/api/retrain")
async def retrain_model_manual(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
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


@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Информация о текущем пользователе"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)