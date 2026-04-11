# stage_01_baseline_model.py
"""
Этап 1: Базовая модель классификации расходов
- Загрузка и предобработка данных
- Векторизация TF-IDF
- Обучение Logistic Regression
- Сохранение модели
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Настройка русского текста в графиках
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class BaselineExpenseClassifier:
    """
    Базовая модель для категоризации расходов
    """
    
    def __init__(self):
        self.pipeline = None
        self.categories = [
            'Продукты', 'Транспорт', 'Кафе', 'Связь', 
            'Коммунальные', 'Развлечения', 'Здоровье', 'Одежда',
            'Такси', 'Аптеки', 'Переводы', 'Другое'
        ]
        self.training_history = {}
        
    def preprocess_text(self, text):
        """
        Предобработка текста транзакции
        
        Алгоритм:
        1. Приведение к нижнему регистру
        2. Удаление цифр (кроме знака рубля)
        3. Удаление пунктуации
        4. Нормализация пробелов
        5. Замена распространённых аббревиатур
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Замена аббревиатур и сленга
        replacements = {
            r'азс': 'заправка',
            r'спб': 'санкт-петербург',
            r'мск': 'москва',
            r'т-банк|тинькофф': 'банк',
            r'сбер|сбербанк': 'банк',
            r'рф|россия': 'россия'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Удаление цифр (но сохраняем валюту)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Удаление пунктуации
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, df):
        """
        Извлечение дополнительных признаков
        """
        df = df.copy()
        
        # Длина текста
        df['text_len'] = df['processed_text'].str.len()
        
        # Количество слов
        df['word_count'] = df['processed_text'].str.split().str.len()
        
        # Признак наличия эмодзи
        df['has_emoji'] = df['description'].str.contains(r'[\U0001F600-\U0001F64F]').astype(int)
        
        # Признак заглавных букв (важность)
        df['is_uppercase'] = (df['description'].str.isupper()).astype(int)
        
        return df
    
    def generate_synthetic_data(self, n_samples=1000):
        """
        Генерация синтетических данных для демонстрации
        """
        np.random.seed(42)
        
        templates = {
            'Продукты': [
                '{}магнит{}', '{}пятерочка{}', '{}ашан{}', '{}перекресток{}',
                '{}дикси{}', '{}продукты{}', '{}еда{}', '{}супермаркет{}',
                '{}овощи{}', '{}фрукты{}', '{}мясо{}', '{}молочка{}'
            ],
            'Транспорт': [
                '{}метро{}', '{}автобус{}', '{}троллейбус{}', '{}трамвай{}',
                '{}электричка{}', '{}проезд{}', '{}транспорт{}', '{}билет{}'
            ],
            'Кафе': [
                '{}кафе{}', '{}ресторан{}', '{}кофейня{}', '{}столовая{}',
                '{}бургер{}', '{}макдоналдс{}', '{}kfc{}', '{}пицца{}',
                '{}суши{}', '{}кофе{}', '{}обед{}', '{}ланч{}'
            ],
            'Такси': [
                '{}яндекс такси{}', '{}uber{}', '{}gett{}', '{}ситимобил{}',
                '{}такси{}', '{}поездка{}'
            ],
            'Аптеки': [
                '{}аптека{}', '{}здоровье{}', '{}фарма{}', '{}лекарство{}',
                '{}витамины{}', '{}аптека.ру{}', '{}таблетки{}'
            ],
            'Связь': [
                '{}мтс{}', '{}билайн{}', '{}мегафон{}', '{}теле2{}',
                '{}интернет{}', '{}связь{}', '{}телефон{}', '{}симкарта{}'
            ],
            'Коммунальные': [
                '{}жку{}', '{}жкх{}', '{}коммуналка{}', '{}свет{}',
                '{}вода{}', '{}газ{}', '{}отопление{}', '{}квартплата{}'
            ],
            'Развлечения': [
                '{}кино{}', '{}театр{}', '{}концерт{}', '{}музей{}',
                '{}игры{}', '{}стим{}', '{}netflix{}', '{}кинотеатр{}'
            ],
            'Одежда': [
                '{}одежда{}', '{}обувь{}', '{}wildberries{}', '{}ozon{}',
                '{}lamoda{}', '{}магазин одежды{}', '{}джинсы{}'
            ],
            'Переводы': [
                '{}перевод{}', '{}перевел{}', '{}отправил{}', '{}получил{}',
                '{}перевод средств{}', '{}перевод на карту{}'
            ]
        }
        
        prefixes = ['', 'оплата ', 'списание ', 'покупка ', 'перевод ']
        suffixes = ['', ' онлайн', ' спб', ' мск', ' *1234']
        
        data = []
        
        for category, tmpl_list in templates.items():
            for _ in range(n_samples // len(templates)):
                tmpl = np.random.choice(tmpl_list)
                prefix = np.random.choice(prefixes)
                suffix = np.random.choice(suffixes)
                description = tmpl.format(prefix, suffix).strip()
                
                # Добавляем шум (опечатки)
                if np.random.random() < 0.2:
                    # Меняем две буквы местами
                    if len(description) > 3:
                        pos = np.random.randint(1, len(description)-1)
                        description = description[:pos] + description[pos+1] + description[pos] + description[pos+2:]
                
                # Добавляем сумму
                amount = -np.random.uniform(100, 5000)
                
                data.append({
                    'description': description,
                    'category': category,
                    'amount': amount
                })
        
        return pd.DataFrame(data)
    
    def train(self, df=None, use_synthetic=True, test_size=0.2):
        """
        Обучение модели
        """
        print("="*60)
        print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
        print("="*60)
        
        # Загружаем или генерируем данные
        if df is None and use_synthetic:
            print("\n1. Генерация синтетических данных...")
            df = self.generate_synthetic_data(2000)
            print(f"   Сгенерировано {len(df)} транзакций")
        elif df is not None:
            print(f"\n1. Загружено {len(df)} транзакций из файла")
        
        print(f"\n2. Распределение по категориям:")
        category_counts = df['category'].value_counts()
        for cat, count in category_counts.items():
            print(f"   {cat}: {count} ({count/len(df)*100:.1f}%)")
        
        # Предобработка текста
        print("\n3. Предобработка текста...")
        df['processed_text'] = df['description'].apply(self.preprocess_text)
        
        # Дополнительные признаки
        df = self.extract_features(df)
        
        # Подготовка данных для обучения
        X_text = df['processed_text']
        y = df['category']
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Обучающая выборка: {len(X_train)}")
        print(f"   Тестовая выборка: {len(X_test)}")
        
        # Создание пайплайна
        print("\n4. Создание ML пайплайна...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,           # Топ-5000 слов
                ngram_range=(1, 2),          # Униграммы + биграммы
                min_df=2,                    # Слово должно встречаться минимум 2 раза
                max_df=0.95,                 # Игнорируем слишком частые слова
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b'
            )),
            ('clf', LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                random_state=42,
                C=1.0,                       # Регуляризация
                class_weight='balanced'      # Учёт дисбаланса классов
            ))
        ])
        
        # Обучение
        print("5. Обучение модели...")
        self.pipeline.fit(X_train, y_train)
        
        # Оценка на тесте
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)
        
        accuracy = (y_pred == y_test).mean()
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("="*60)
        print(f"Точность (Accuracy): {accuracy:.3f}")
        print(f"F1-score (macro): {f1_macro:.3f}")
        
        # Детальный отчёт
        print("\nДетальный отчёт по категориям:")
        print(classification_report(y_test, y_pred))
        
        # Сохраняем историю обучения
        self.training_history = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'classes': self.pipeline.classes_,
            'n_samples': len(df)
        }
        
        # Сохраняем предсказания для анализа ошибок
        self._analyze_errors(X_test, y_test, y_pred, y_proba)
        
        return self.pipeline
    
    def _analyze_errors(self, X_test, y_test, y_pred, y_proba):
        """
        Анализ ошибок модели
        """
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred, labels=self.pipeline.classes_)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.pipeline.classes_,
                    yticklabels=self.pipeline.classes_)
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.show()
        
        # Находим самые неуверенные предсказания
        max_proba = np.max(y_proba, axis=1)
        uncertain_indices = np.argsort(max_proba)[:10]
        
        print("\nСамые неуверенные предсказания:")
        for idx in uncertain_indices:
            print(f"  Текст: '{X_test.iloc[idx]}'")
            print(f"  Истинная: {y_test.iloc[idx]}")
            print(f"  Предсказанная: {y_pred[idx]}")
            print(f"  Уверенность: {max_proba[idx]:.3f}")
            print()
    
    def predict(self, description, amount=None):
        """
        Предсказание категории для одной транзакции
        """
        if self.pipeline is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Предобработка
        processed = self.preprocess_text(description)
        
        # Предсказание
        pred = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        confidence = max(proba)
        
        # Логика для переводов на основе суммы
        if amount and abs(amount) > 10000:
            if any(word in description.lower() for word in ['перевод', 'перевел', 'отправил']):
                return 'Переводы', confidence
        
        return pred, confidence
    
    def predict_batch(self, texts):
        """
        Пакетное предсказание
        """
        processed = [self.preprocess_text(t) for t in texts]
        predictions = self.pipeline.predict(processed)
        confidences = np.max(self.pipeline.predict_proba(processed), axis=1)
        
        return list(zip(predictions, confidences))
    
    def save_model(self, path='baseline_model.pkl'):
        """
        Сохранение модели
        """
        model_data = {
            'pipeline': self.pipeline,
            'categories': self.categories,
            'training_history': self.training_history
        }
        joblib.dump(model_data, path)
        print(f"\nМодель сохранена в {path}")
    
    def load_model(self, path='baseline_model.pkl'):
        """
        Загрузка модели
        """
        model_data = joblib.load(path)
        self.pipeline = model_data['pipeline']
        self.categories = model_data['categories']
        self.training_history = model_data['training_history']
        print(f"Модель загружена из {path}")
        print(f"Точность модели при обучении: {self.training_history['accuracy']:.3f}")


def demo():
    """
    Демонстрация работы модели
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ МОДЕЛИ")
    print("="*60)
    
    # Инициализация и обучение
    classifier = BaselineExpenseClassifier()
    classifier.train(use_synthetic=True)
    
    # Сохранение модели
    classifier.save_model()
    
    # Тестирование на новых примерах
    test_transactions = [
        ("Пятерочка 1234", None),
        ("Яндекс Такси поездка", 300),
        ("Макдоналдс Пушкинская", 450),
        ("Перевод на карту Ивану", 15000),
        ("Аптека 36.6 на Ленина", 567),
        ("МТС интернет 500руб", 500),
        ("ЖКУ за январь", 3500),
        ("Кинотеатр Аврора", 600),
        ("Wildberries заказ", 2500),
        ("Метро проезд", 50),
    ]
    
    print("\n" + "="*60)
    print("ПРЕДСКАЗАНИЯ ДЛЯ НОВЫХ ТРАНЗАКЦИЙ")
    print("="*60)
    
    results = []
    for desc, amount in test_transactions:
        category, confidence = classifier.predict(desc, amount)
        results.append({
            'description': desc,
            'amount': amount if amount else 'не указана',
            'predicted': category,
            'confidence': f"{confidence:.2%}"
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Анализ уверенности по категориям
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО МОДЕЛИ")
    print("="*60)
    print(f"Количество категорий: {len(classifier.categories)}")
    print(f"Категории: {', '.join(classifier.categories)}")
    print(f"Размер TF-IDF словаря: {classifier.pipeline.named_steps['tfidf'].get_feature_names_out().shape[0]}")
    
    return classifier


if __name__ == "__main__":
    # Запуск демонстрации
    model = demo()
    
    # Интерактивный режим
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Введите описание транзакции (или 'quit' для выхода)")
    print("="*60)
    
    while True:
        user_input = input("\nОписание: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        category, confidence = model.predict(user_input)
        print(f"Категория: {category} (уверенность: {confidence:.2%})")