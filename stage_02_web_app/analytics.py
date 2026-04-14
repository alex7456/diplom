"""
Сервис аналитики и построения графиков
"""

from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd
from database import Transaction, User


class AnalyticsService:
    """Сервис для аналитики расходов"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_expenses_by_category(self, user_id: int, days: int = 30) -> Dict[str, float]:
        """Расходы по категориям за последние N дней"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.created_at >= since_date,
            Transaction.amount < 0
        ).all()
        
        category_expenses = defaultdict(float)
        for t in transactions:
            category = t.get_final_category()
            category_expenses[category] += abs(t.amount)
        
        return dict(category_expenses)
    
    def get_monthly_trend(self, user_id: int, months: int = 6) -> Dict[str, Dict]:
        """Динамика расходов по месяцам"""
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.amount < 0
        ).all()
        
        monthly_data = defaultdict(lambda: defaultdict(float))
        
        for t in transactions:
            month_key = t.created_at.strftime('%Y-%m')
            category = t.get_final_category()
            monthly_data[month_key][category] += abs(t.amount)
        
        sorted_months = sorted(monthly_data.keys())[-months:]
        
        result = {}
        for month in sorted_months:
            result[month] = dict(monthly_data[month])
        
        return result
    
    def get_top_merchants(self, user_id: int, limit: int = 10) -> List[Tuple[str, float, int]]:
        """Топ магазинов по сумме трат"""
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.amount < 0
        ).all()
        
        merchant_stats = defaultdict(lambda: {'total': 0, 'count': 0})
        
        for t in transactions:
            merchant = t.description.split()[0] if t.description else "unknown"
            merchant_stats[merchant]['total'] += abs(t.amount)
            merchant_stats[merchant]['count'] += 1
        
        sorted_merchants = sorted(
            merchant_stats.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:limit]
        
        return [(name, stats['total'], stats['count']) for name, stats in sorted_merchants]
    
    def get_daily_spending(self, user_id: int, days: int = 30) -> Dict[str, float]:
        """Ежедневные расходы"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.created_at >= since_date,
            Transaction.amount < 0
        ).all()
        
        daily = defaultdict(float)
        for t in transactions:
            day_key = t.created_at.strftime('%Y-%m-%d')
            daily[day_key] += abs(t.amount)
        
        return dict(daily)
    
    def get_category_breakdown(self, user_id: int) -> Dict:
        """Детальная разбивка по категориям"""
        stats = {
            'total_expenses': 0,
            'total_transactions': 0,
            'categories': {},
            'average_per_transaction': 0
        }
        
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.amount < 0
        ).all()
        
        if not transactions:
            return stats
        
        total_expenses = 0
        category_data = defaultdict(lambda: {'total': 0, 'count': 0})
        
        for t in transactions:
            category = t.get_final_category()
            amount = abs(t.amount)
            total_expenses += amount
            category_data[category]['total'] += amount
            category_data[category]['count'] += 1
        
        stats['total_expenses'] = total_expenses
        stats['total_transactions'] = len(transactions)
        stats['average_per_transaction'] = total_expenses / len(transactions) if transactions else 0
        
        for cat, data in category_data.items():
            stats['categories'][cat] = {
                'total': data['total'],
                'count': data['count'],
                'percentage': (data['total'] / total_expenses * 100) if total_expenses > 0 else 0
            }
        
        return stats
    
    def get_export_data(self, user_id: int, days: int = 90) -> pd.DataFrame:
        """Получение данных для экспорта"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.created_at >= since_date
        ).order_by(Transaction.created_at.desc()).all()
        
        data = []
        for t in transactions:
            data.append({
                'Дата': t.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'Описание': t.description,
                'Сумма': t.amount,
                'Категория': t.get_final_category(),
                'Предсказанная категория': t.predicted_category,
                'Уверенность': f"{t.confidence * 100:.1f}%",
                'Примечания': t.notes or ''
            })
        
        return pd.DataFrame(data)
    
    def get_analytics_by_source(self, user_id: int, is_auto: bool) -> dict:
        """Аналитика по источнику данных (ручной ввод или выписка)"""
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.is_auto == is_auto
        ).all()
        
        total_expenses = sum(abs(t.amount) for t in transactions if t.amount and t.amount < 0)
        total_income = sum(t.amount for t in transactions if t.amount and t.amount > 0)
        
        category_stats = {}
        for t in transactions:
            if t.amount and t.amount < 0:
                category = t.get_final_category()
                amount = abs(t.amount)
                if category not in category_stats:
                    category_stats[category] = {'count': 0, 'total': 0}
                category_stats[category]['count'] += 1
                category_stats[category]['total'] += amount
        
        # Динамика по месяцам
        monthly = {}
        for t in transactions:
            if t.amount and t.amount < 0:
                month_key = t.created_at.strftime('%Y-%m')
                category = t.get_final_category()
                if month_key not in monthly:
                    monthly[month_key] = {}
                monthly[month_key][category] = monthly[month_key].get(category, 0) + abs(t.amount)
        
        # Топ магазинов (для выписок)
        top_merchants = []
        if not is_auto:
            merchant_stats = {}
            for t in transactions:
                if t.amount and t.amount < 0:
                    merchant = t.description.split()[0] if t.description else "unknown"
                    merchant_stats[merchant] = merchant_stats.get(merchant, 0) + abs(t.amount)
            top_merchants = [{"name": k, "total": v} for k, v in sorted(merchant_stats.items(), key=lambda x: x[1], reverse=True)[:8]]
        
        # Ежедневные расходы (для выписок)
        daily = {}
        if not is_auto:
            for t in transactions:
                if t.amount and t.amount < 0:
                    day_key = t.created_at.strftime('%Y-%m-%d')
                    daily[day_key] = daily.get(day_key, 0) + abs(t.amount)
        
        return {
            'total_expenses': total_expenses,
            'total_income': total_income,
            'total_transactions': len(transactions),
            'categories': category_stats,
            'monthly': monthly,
            'top_merchants': top_merchants,
            'daily': daily
        }