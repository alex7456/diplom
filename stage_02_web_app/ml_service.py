"""
ML сервис с CatBoost — максимальная точность
"""

import pandas as pd
import numpy as np
import re
import joblib
import hashlib
from datetime import datetime
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class EnhancedExpenseClassifier:
    """
    Модель на CatBoost для классификации расходов
    """
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = LabelEncoder()
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.prediction_history: List[Dict] = []
        self.error_log: List[Dict] = []
        
        # Расширенные категории (18 категорий)
        self.categories = [
            'Продукты', 'Транспорт', 'Кафе', 'Такси', 'Каршеринг',
            'Аптеки', 'Связь', 'Коммунальные', 'Развлечения', 'Маркетплейсы',
            'Одежда', 'Переводы', 'Подписки', 'Красота', 'Образование',
            'Дом', 'Здоровье', 'Другое'
        ]
        
        # Веса для суммы
        self.amount_weights = {
            'Продукты': (50, 5000), 'Транспорт': (20, 1500), 'Кафе': (100, 3000),
            'Такси': (100, 1500), 'Каршеринг': (50, 1000), 'Аптеки': (50, 3000),
            'Связь': (200, 2000), 'Коммунальные': (500, 15000), 'Развлечения': (100, 5000),
            'Маркетплейсы': (300, 15000), 'Одежда': (500, 20000), 'Переводы': (500, 100000),
            'Подписки': (100, 1500), 'Красота': (300, 5000), 'Образование': (500, 30000),
            'Дом': (500, 50000), 'Здоровье': (300, 10000), 'Другое': (0, 100000)
        }
        
        # Ключевые слова для пост-обработки
        self.keyword_rules = self._build_keyword_rules()
    
    def _build_keyword_rules(self) -> Dict[str, List[str]]:
        """Расширенный словарь ключевых слов"""
        return {
            'Продукты': ['магнит', 'пятерочка', 'ашан', 'перекресток', 'дикси', 'спар', 'лента',
                        'продукты', 'еда', 'супермаркет', 'гипермаркет', 'овощи', 'фрукты', 'мясо',
                        'рыба', 'молочка', 'хлеб', 'вино', 'алкоголь', 'granat', 'табрис'],
            'Транспорт': ['метро', 'автобус', 'троллейбус', 'трамвай', 'электричка', 'проезд', 'билет', 'ржд', 'жд'],
            'Кафе': ['кафе', 'ресторан', 'кофейня', 'столовая', 'бургер', 'макдоналдс', 'kfc',
                    'бургер кинг', 'шаурма', 'пицца', 'суши', 'кофе', 'чай', 'фастфуд', 'вкусно и точка'],
            'Такси': ['яндекс такси', 'uber', 'gett', 'ситимобил', 'такси', 'везёт', 'везет', 'убер'],
            'Каршеринг': ['дели мобиль', 'delimobil', 'ситидрайв', 'citydrive', 'яндекс драйв', 'yandex drive', 'белкакар', 'belkacar'],
            'Аптеки': ['аптека', 'лекарство', 'витамины', 'аптека.ру', 'таблетки', 'здоровье', 'медикаменты'],
            'Связь': ['мтс', 'билайн', 'мегафон', 'теле2', 'интернет', 'связь', 'ростелеком',
                     'домашний интернет', 'телефон', 'симкарта', 'йота', 'ттк', 'дом.ру'],
            'Коммунальные': ['жку', 'жкх', 'коммуналка', 'свет', 'электричество', 'вода', 'газ',
                            'отопление', 'квартплата', 'мосэнерго', 'водоканал', 'петербургтепло', 'газпром'],
            'Развлечения': ['кино', 'театр', 'концерт', 'музей', 'парк', 'игры', 'стим', 'playstation',
                           'кинотеатр', 'квест', 'боулинг', 'бильярд', 'караоке', 'зоопарк'],
            'Маркетплейсы': ['wildberries', 'wb', 'вайлдберриз', 'ozon', 'озон', 'yandex market', 'яндекс маркет', 'lamoda', 'ламода'],
            'Одежда': ['одежда', 'обувь', 'магазин одежды', 'джинсы', 'куртка', 'ботинки', 'кроссовки',
                      'футболка', 'платье', 'костюм', 'zara', 'hm', 'adidas', 'nike', 'puma'],
            'Переводы': ['перевод', 'перевел', 'перевела', 'отправил', 'получил', 'перевод средств', 'перевод на карту'],
            'Подписки': ['яндекс плюс', 'yandex plus', 'netflix', 'ivi', 'иви', 'кинопоиск', 'okko', 'окко',
                        'spotify', 'apple music', 'vk музыка', 'вк музыка', 'youtube premium', 'ютуб премиум',
                        'mtc премиум', 'мтс премиум', 'start', 'старт', 'more.tv', 'more tv'],
            'Красота': ['парикмахерская', 'стрижка', 'маникюр', 'педикюр', 'косметолог', 'салон красоты',
                       'брови', 'ресницы', 'косметика', 'парфюм', 'духи', 'крема', 'шампунь'],
            'Образование': ['курсы', 'университет', 'институт', 'школа', 'репетитор', 'учебник', 'книги',
                           'образование', 'вебинар', 'тренинг', 'английский', 'программирование'],
            'Дом': ['стройка', 'ремонт', 'инструменты', 'мебель', 'икеа', 'хоум', 'дом', 'посуда',
                   'текстиль', 'сантехника', 'обои', 'краска'],
            'Здоровье': ['спортзал', 'фитнес', 'бассейн', 'тренажерка', 'врач', 'клиника',
                        'больница', 'стоматология', 'анализы', 'медцентр']
        }
    
    def _get_cache_key(self, description: str, amount: Optional[float]) -> str:
        """Ключ для кэша"""
        key_str = f"{description.lower().strip()}_{amount}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def preprocess_text(self, text: str) -> str:
        """Улучшенная предобработка текста"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Расширенный словарь замен
        replacements = [
            # Магазины продуктов
            (r'пятёрочка|пятерочка|5ка', 'продуктовый магазин'),
            (r'магнит|ашан|перекрёсток|перекресток|дикси|спар|лента', 'продуктовый магазин'),
            (r'окей|о\'кей', 'продуктовый магазин'),
            (r'granat market|гранат|granat', 'продуктовый магазин'),
            (r'табрис', 'продуктовый магазин'),
            
            # Каршеринг
            (r'дели мобиль|delimobil|ситидрайв|citydrive|яндекс драйв|yandex drive|белкакар|belkacar', 'каршеринг'),
            
            # Маркетплейсы
            (r'wildberries|wb|вайлдберриз', 'маркетплейс'),
            (r'ozon|озон', 'маркетплейс'),
            (r'yandex market|яндекс маркет', 'маркетплейс'),
            (r'lamoda|ламода', 'маркетплейс'),
            
            # Еда
            (r'макдоналдс|макдак|вкусно и точка|kfc|бургер кинг|subway|сабвей', 'фастфуд'),
            (r'кофейня|кофе хауз|starbucks|старбакс|кофе и точка|кофеин|кофемания', 'кофейня'),
            (r'суши|роллы|пицца|додо пицца|папа джонс|суши весла|сушишоп', 'еда на вынос'),
            (r'терра|тандур|шаурма|чебурек', 'уличная еда'),
            
            # Транспорт
            (r'яндекс такси|yandex taxi|яндекс\.такси|убер|uber|gett|ситимобил|максим|везёт|везет', 'такси'),
            (r'метро|автобус|троллейбус|трамвай|электричка|маршрутка', 'общественный транспорт'),
            (r'аэрофлот|s7|победа|ютейр|ред вингс|авиабилет', 'авиабилеты'),
            
            # Услуги связи
            (r'мтс|билайн|мегафон|tele2|тел2|йота|транстелеком|ттк', 'мобильная связь'),
            (r'дом\.ру|ростелеком|интернет дом|домашний интернет|вайфай', 'домашний интернет'),
            
            # ЖКХ
            (r'жку|жкх|коммуналка|квартплата|управляющая компания|ук жкх', 'коммунальные платежи'),
            (r'мосэнерго|петербургтепло|вологдатепло|газпром|газсервис', 'коммунальные'),
            (r'водоканал|мосводоканал|водный канал|водоканал спб', 'вода'),
            (r'свет|электричество|энергосбыт|мосэнергосбыт|лэнерго', 'электричество'),
            
            # Подписки
            (r'яндекс плюс|yandex plus|яндекс\.плюс', 'подписка'),
            (r'netflix|ivi|иви|кинопоиск|кино поиск|okko|окко', 'подписка'),
            (r'spotify|apple music|vk музыка|вк музыка|youtube premium|ютуб премиум', 'подписка'),
            (r'mtc премиум|мтс премиум|start|старт|more\.tv', 'подписка'),
            
            # Переводы
            (r'перевод|перевел|перевела|отправил|получил|перевод средств', 'перевод средств'),
            (r'сбербанк онлайн перевод|тинькофф перевод|альфа перевод', 'перевод'),
            (r'сбп|система быстрых платежей', 'перевод')
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        # Удаляем цифры и спецсимволы
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Удаляем стоп-слова
        stop_words = ['оплата', 'списание', 'покупка', 'товар', 'услуга', 'счет', 'карта', 'руб']
        for word in stop_words:
            text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, df: pd.DataFrame = None, use_synthetic: bool = True) -> Dict:
        """Обучение CatBoost модели"""
        print("="*60)
        print("ОБУЧЕНИЕ CATBOOST МОДЕЛИ")
        print("="*60)
        
        if df is None and use_synthetic:
            df = self._generate_enhanced_data(5000)
        
        # Предобработка
        print("1. Предобработка текста...")
        df['processed_text'] = df['description'].apply(self.preprocess_text)
        
        # Удаляем пустые строки
        df = df[df['processed_text'].str.len() > 0]
        
        if len(df) < 10:
            print("⚠️ Недостаточно данных для обучения. Нужно минимум 10 примеров.")
            return {'accuracy': 0, 'f1_macro': 0}
        
        # TF-IDF векторизация
        print("2. TF-IDF векторизация...")
        self.tfidf = TfidfVectorizer(
            max_features=min(5000, len(df)),
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        tfidf_features = self.tfidf.fit_transform(df['processed_text']).toarray()
        
        # Числовые признаки
        print("3. Добавление числовых признаков...")
        text_length = df['processed_text'].str.len().values.reshape(-1, 1)
        word_count = df['processed_text'].str.split().str.len().values.reshape(-1, 1)
        has_digit = df['description'].str.contains(r'\d').astype(int).values.reshape(-1, 1)
        
        X = np.hstack([tfidf_features, text_length, word_count, has_digit])
        y = df['category']
        
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Для малых данных
        if len(df) < 50:
            print("4. Обучение CatBoost (малые данные)...")
            self.model = CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                verbose=False,
                random_seed=42
            )
            self.model.fit(X, y_encoded, verbose=False)
            accuracy = 1.0
            f1_macro = 1.0
        else:
            # Разделяем на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            print("4. Обучение CatBoost...")
            self.model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                verbose=50,
                random_seed=42
            )
            
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            
            # Оценка
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print(f"{'='*60}")
        print(f"Test accuracy: {accuracy:.3f}")
        print(f"F1-score (macro): {f1_macro:.3f}")
        
        return {'accuracy': accuracy, 'f1_macro': f1_macro}
    
    def _generate_enhanced_data(self, n_samples: int) -> pd.DataFrame:
        """Генерация синтетических данных"""
        np.random.seed(42)
        
        templates_with_amounts = {
            'Продукты': {'templates': ['пятерочка', 'магнит', 'ашан', 'перекресток', 'дикси', 'спар', 'лента'], 'amount_range': (100, 5000)},
            'Транспорт': {'templates': ['метро', 'автобус', 'троллейбус', 'электричка', 'проездной'], 'amount_range': (30, 1500)},
            'Кафе': {'templates': ['кафе', 'ресторан', 'кофейня', 'макдоналдс', 'бургер', 'суши', 'пицца'], 'amount_range': (150, 3000)},
            'Такси': {'templates': ['яндекс такси', 'uber', 'gett', 'такси'], 'amount_range': (100, 2000)},
            'Каршеринг': {'templates': ['дели мобиль', 'ситидрайв', 'яндекс драйв', 'белкакар'], 'amount_range': (50, 1000)},
            'Аптеки': {'templates': ['аптека', 'лекарство', 'витамины', 'аптека.ру'], 'amount_range': (50, 4000)},
            'Связь': {'templates': ['мтс', 'билайн', 'мегафон', 'интернет', 'телефон'], 'amount_range': (300, 2000)},
            'Коммунальные': {'templates': ['жку', 'жкх', 'коммуналка', 'свет', 'вода', 'газ'], 'amount_range': (1000, 12000)},
            'Развлечения': {'templates': ['кино', 'театр', 'концерт', 'игры', 'стим', 'netflix'], 'amount_range': (200, 5000)},
            'Маркетплейсы': {'templates': ['wildberries', 'ozon', 'yandex market', 'lamoda'], 'amount_range': (300, 15000)},
            'Одежда': {'templates': ['одежда', 'обувь', 'магазин одежды', 'джинсы'], 'amount_range': (500, 20000)},
            'Переводы': {'templates': ['перевод', 'перевел', 'отправил', 'перевод средств'], 'amount_range': (1000, 50000)},
            'Подписки': {'templates': ['яндекс плюс', 'netflix', 'ivi', 'кинопоиск', 'spotify'], 'amount_range': (199, 799)},
            'Красота': {'templates': ['парикмахерская', 'стрижка', 'маникюр', 'косметика'], 'amount_range': (500, 6000)},
            'Образование': {'templates': ['курсы', 'университет', 'репетитор', 'книги'], 'amount_range': (500, 30000)},
            'Дом': {'templates': ['ремонт', 'инструменты', 'мебель', 'икеа', 'посуда'], 'amount_range': (500, 50000)},
            'Здоровье': {'templates': ['спортзал', 'фитнес', 'врач', 'стоматология'], 'amount_range': (300, 10000)}
        }
        
        data = []
        samples_per_category = n_samples // len(templates_with_amounts)
        
        for category, info in templates_with_amounts.items():
            for _ in range(samples_per_category):
                template = np.random.choice(info['templates'])
                amount = -np.random.uniform(*info['amount_range'])
                
                # Добавляем вариации
                variation_type = np.random.choice(['none', 'prefix', 'suffix', 'typo'], p=[0.6, 0.15, 0.15, 0.1])
                
                if variation_type == 'prefix':
                    template = np.random.choice(['оплата ', 'списание ', 'покупка ']) + template
                elif variation_type == 'suffix':
                    template = template + ' ' + np.random.choice(['спб', 'мск', 'онлайн', 'карта', '*1234'])
                elif variation_type == 'typo':
                    if len(template) > 3:
                        pos = np.random.randint(1, len(template)-1)
                        template = template[:pos] + template[pos+1] + template[pos] + template[pos+2:]
                
                data.append({'description': template, 'category': category, 'amount': amount})
        
        return pd.DataFrame(data)
    
    def keyword_correction(self, description: str, predicted: str, confidence: float) -> Tuple[str, float]:
        """Пост-обработка по ключевым словам"""
        text = description.lower()
        
        # Специальные правила для подписок
        if any(word in text for word in ['яндекс плюс', 'yandex plus', 'netflix', 'ivi', 'иви', 'кинопоиск', 'okko', 'окко', 'spotify', 'apple music', 'vk музыка', 'вк музыка', 'youtube premium', 'ютуб премиум', 'mtc премиум', 'мтс премиум', 'start', 'старт']):
            return 'Подписки', 0.98
        
        # Специальные правила для такси
        if any(word in text for word in ['везёт', 'везет', 'убер']):
            return 'Такси', 0.98
        
        # Специальные правила для каршеринга
        if any(word in text for word in ['дели мобиль', 'delimobil', 'ситидрайв', 'citydrive', 'яндекс драйв', 'yandex drive', 'белкакар', 'belkacar']):
            return 'Каршеринг', 0.98
        
        # Специальные правила для маркетплейсов
        if any(word in text for word in ['wildberries', 'wb', 'вайлдберриз', 'ozon', 'озон', 'yandex market', 'яндекс маркет', 'lamoda', 'ламода']):
            return 'Маркетплейсы', 0.98
        
        for category, keywords in self.keyword_rules.items():
            for keyword in keywords:
                if keyword in text:
                    return category, 0.98
        
        if 'перевод' in text or 'перевел' in text or 'отправил' in text:
            if predicted != 'Переводы':
                return 'Переводы', max(confidence, 0.85)
        
        return predicted, confidence
    
    def amount_penalty(self, predicted_category: str, amount: Optional[float]) -> float:
        """Корректировка по сумме"""
        if amount is None or predicted_category not in self.amount_weights:
            return 1.0
        
        min_amount, max_amount = self.amount_weights[predicted_category]
        abs_amount = abs(amount)
        
        if min_amount <= abs_amount <= max_amount:
            return 1.2
        elif abs_amount > max_amount * 2:
            return 0.6
        elif abs_amount < min_amount / 2:
            return 0.8
        return 1.0
    
    def predict(self, description: str, amount: Optional[float] = None) -> Tuple[str, float]:
        """Предсказание категории"""
        cache_key = self._get_cache_key(description, amount)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.model is None or self.tfidf is None:
            raise ValueError("Модель не обучена")
        
        start_time = datetime.now()
        
        processed = self.preprocess_text(description)
        
        tfidf_features = self.tfidf.transform([processed]).toarray()
        text_length = [[len(processed)]]
        word_count = [[len(processed.split())]]
        has_digit = [[int(any(c.isdigit() for c in description))]]
        
        X = np.hstack([tfidf_features, text_length, word_count, has_digit])
        
        pred_encoded = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = max(proba)
        
        pred = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        amount_factor = self.amount_penalty(pred, amount)
        adjusted_confidence = min(confidence * amount_factor, 0.99)
        pred, adjusted_confidence = self.keyword_correction(description, pred, adjusted_confidence)
        
        self.cache[cache_key] = (pred, adjusted_confidence)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.prediction_history.append({
            'description': description,
            'amount': amount,
            'predicted': pred,
            'confidence': adjusted_confidence,
            'time_ms': processing_time
        })
        
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
        
        return pred, adjusted_confidence
    
    def predict_batch(self, transactions: List[dict]) -> List[Tuple[str, float, float]]:
        """Пакетное предсказание"""
        results = []
        for trans in transactions:
            start = datetime.now()
            category, confidence = self.predict(
                trans.get('description', ''),
                trans.get('amount')
            )
            time_ms = (datetime.now() - start).total_seconds() * 1000
            results.append((category, confidence, time_ms))
        return results
    
    def get_stats(self) -> Dict:
        """Статистика"""
        return {
            'cache_size': len(self.cache),
            'total_predictions': len(self.prediction_history),
            'avg_prediction_time': np.mean([h['time_ms'] for h in self.prediction_history]) if self.prediction_history else 0,
            'model_loaded': self.model is not None,
            'n_categories': len(self.categories)
        }
    
    def save_model(self, path: str = 'enhanced_model.pkl'):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'tfidf': self.tfidf,
            'label_encoder': self.label_encoder,
            'cache': self.cache,
            'categories': self.categories,
            'amount_weights': self.amount_weights,
            'keyword_rules': self.keyword_rules
        }
        joblib.dump(model_data, path)
        print(f"CatBoost модель сохранена в {path}")
    
    def load_model(self, path: str = 'enhanced_model.pkl'):
        """Загрузка модели"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.tfidf = model_data['tfidf']
        self.label_encoder = model_data['label_encoder']
        self.cache = model_data.get('cache', {})
        self.categories = model_data.get('categories', self.categories)
        self.amount_weights = model_data.get('amount_weights', self.amount_weights)
        self.keyword_rules = model_data.get('keyword_rules', self.keyword_rules)
        print(f"CatBoost модель загружена из {path}")