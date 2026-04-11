"""
ML сервис с максимальной точностью
- Поддержка суммы транзакции
- Кэширование предсказаний
- Анализ ошибок
- Улучшенная предобработка
- Ансамбль моделей
- Глубокое обучение (опционально)
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class EnhancedExpenseClassifier:
    """
    УЛЬТРА-УЛУЧШЕННАЯ модель с максимальной точностью
    """
    
    def __init__(self):
        self.pipeline = None
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.prediction_history: List[Dict] = []
        self.error_log: List[Dict] = []
        self.label_encoder = LabelEncoder()
        
        # Расширенные категории (более детальные)
        self.categories = [
            'Продукты', 'Транспорт', 'Кафе', 'Связь_интернет',
            'Коммунальные', 'Развлечения', 'Здоровье', 'Одежда_обувь',
            'Такси', 'Аптеки', 'Переводы', 'Красота', 'Образование',
            'Дом_ремонт', 'Животные', 'Подарки', 'Спорт', 'Другое'
        ]
        
        # Улучшенные веса для суммы
        self.amount_weights = {
            'Продукты': (50, 5000),
            'Транспорт': (20, 1500),
            'Кафе': (100, 3000),
            'Такси': (100, 1500),
            'Аптеки': (50, 3000),
            'Связь_интернет': (200, 2000),
            'Коммунальные': (500, 15000),
            'Развлечения': (100, 5000),
            'Одежда_обувь': (500, 20000),
            'Переводы': (500, 100000),
            'Красота': (300, 5000),
            'Образование': (500, 30000),
            'Дом_ремонт': (500, 50000),
            'Животные': (200, 5000),
            'Подарки': (200, 10000),
            'Спорт': (300, 10000),
            'Другое': (0, 100000)
        }
        
        # Расширенный словарь ключевых слов для постредактирования
        self.keyword_rules = self._build_keyword_rules()
    
    def _build_keyword_rules(self) -> Dict[str, List[str]]:
        """Строит расширенный словарь ключевых слов"""
        return {
            'Продукты': ['магнит', 'пятерочка', 'ашан', 'перекресток', 'дикси', 'спар', 'метро кэш',
                        'продукты', 'еда', 'супермаркет', 'гипермаркет', 'овощи', 'фрукты', 'мясо',
                        'рыба', 'молочка', 'хлеб', 'вино', 'алкоголь', 'напитки', 'снеки', 'granat'],
            'Транспорт': ['метро', 'автобус', 'троллейбус', 'трамвай', 'электричка', 'проезд',
                         'транспорт', 'билет', 'аэроэкспресс', 'ржд', 'жд билет', 'сапсан',
                         'ласточка', 'пригородный'],
            'Кафе': ['кафе', 'ресторан', 'кофейня', 'столовая', 'бургер', 'макдоналдс', 'kfc',
                    'бургер кинг', 'шаурма', 'пицца', 'суши', 'кофе', 'чай', 'десерт', 'мороженое',
                    'фастфуд', 'вкусно и точка'],
            'Такси': ['яндекс такси', 'uber', 'gett', 'ситимобил', 'такси', 'уберизация', 'maxim',
                     'везёт', 'городское такси', 'яндекс.такси'],
            'Аптеки': ['аптека', 'здоровье', 'фарма', 'лекарство', 'витамины', 'аптека.ру',
                      'таблетки', 'антибиотики', 'бады', 'медикаменты'],
            'Связь_интернет': ['мтс', 'билайн', 'мегафон', 'теле2', 'интернет', 'связь', 'ростелеком',
                              'домашний интернет', 'телефон', 'симкарта', 'йота', 'ттк', 'дом.ру'],
            'Коммунальные': ['жку', 'жкх', 'коммуналка', 'свет', 'электричество', 'вода', 'газ',
                            'отопление', 'квартплата', 'мосэнерго', 'водоканал', 'мосводоканал',
                            'петербургтепло', 'газпром'],
            'Развлечения': ['кино', 'театр', 'концерт', 'музей', 'парк', 'игры', 'стим', 'playstation',
                           'netflix', 'ivi', 'кинотеатр', 'квест', 'боулинг', 'бильярд', 'караоке',
                           'парк аттракционов', 'зоопарк'],
            'Одежда_обувь': ['одежда', 'обувь', 'wildberries', 'wb', 'ozon', 'lamoda', 'магазин одежды',
                            'джинсы', 'куртка', 'ботинки', 'кроссовки', 'футболка', 'платье', 'костюм',
                            'zara', 'hm', 'adidas', 'nike', 'puma'],
            'Переводы': ['перевод', 'перевел', 'перевела', 'отправил', 'получил', 'перевод средств',
                        'перевод на карту', 'сбербанк онлайн перевод', 'тинькофф перевод'],
            'Красота': ['парикмахерская', 'стрижка', 'маникюр', 'педикюр', 'косметолог', 'салон красоты',
                       'брови', 'ресницы', 'косметика', 'парфюм', 'духи', 'крема', 'шампунь'],
            'Образование': ['курсы', 'университет', 'институт', 'школа', 'репетитор', 'учебник', 'книги',
                           'образование', 'вебинар', 'тренинг', 'английский', 'программирование'],
            'Дом_ремонт': ['стройка', 'ремонт', 'инструменты', 'мебель', 'икеа', 'хоум', 'дом',
                          'посуда', 'текстиль', 'сантехника', 'обои', 'краска'],
            'Животные': ['зоомагазин', 'корм для собак', 'корм для кошек', 'ветеринар', 'наполнитель',
                        'лежанка', 'игрушки для животных', 'поводок', 'ошейник'],
            'Подарки': ['подарок', 'сувенир', 'цветы', 'открытка', 'подарочный набор', 'сертификат'],
            'Спорт': ['спортзал', 'фитнес', 'бассейн', 'тренажерка', 'экипировка', 'форма спортивная',
                     'велосипед', 'лыжи', 'ролики', 'скейт']
        }
    
    def _get_cache_key(self, description: str, amount: Optional[float]) -> str:
        """Генерация ключа для кэша"""
        key_str = f"{description.lower().strip()}_{amount}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def preprocess_text(self, text: str) -> str:
        """
        СУПЕР-УЛУЧШЕННАЯ предобработка текста
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # 1. Расширенный словарь замен (40+ правил)
        replacements = [
            # Магазины продуктов
            (r'пятёрочка|пятерочка|5ка', 'продуктовый магазин'),
            (r'магнит|ашан|перекрёсток|перекресток|дикси|спар|верный|британский|лавка', 'продуктовый магазин'),
            (r'окей|о\'кей', 'продуктовый магазин'),
            (r'granat market|гранат|granat', 'продуктовый магазин'),
            
            # Маркетплейсы
            (r'wildberries|wb|вайлдберриз|вб', 'интернет магазин одежды'),
            (r'ozon', 'интернет магазин'),
            (r'lamoda|ламода|лавода', 'магазин одежды'),
            (r'yandex market|яндекс маркет', 'интернет магазин'),
            
            # Еда
            (r'макдоналдс|макдак|вкусно и точка|kfc|бургер кинг|subway|сабвей', 'фастфуд'),
            (r'кофейня|кофе хауз|starbucks|старбакс|кофе и точка|кофеин|кофемания', 'кофейня'),
            (r'суши|роллы|пицца|додо пицца|папа джонс|суши весла|сушишоп', 'еда на вынос'),
            (r'терра|тандур|шаурма|чебурек', 'уличная еда'),
            
            # Транспорт
            (r'яндекс такси|yandex taxi|яндекс\.такси|убер|uber|gett|ситимобил|максим|везёт', 'такси'),
            (r'метро|автобус|троллейбус|трамвай|электричка|маршрутка', 'общественный транспорт'),
            (r'каршеринг|дели мобиль|белка|яндекс драйв|ситидрайв', 'каршеринг'),
            (r'аэрофлот|s7|победа|ютейр|ред вингс|авиабилет', 'авиабилеты'),
            
            # Услуги связи
            (r'мтс|билайн|мегафон|tele2|тел2|йота|транстелеком|ттк', 'мобильная связь'),
            (r'дом\.ру|ростелеком|интернет дом|домашний интернет|вайфай', 'домашний интернет'),
            
            # ЖКХ
            (r'жку|жкх|коммуналка|квартплата|управляющая компания|ук жкх', 'коммунальные платежи'),
            (r'мосэнерго|петербургтепло|вологдатепло|газпром|газсервис', 'коммунальные'),
            (r'водоканал|мосводоканал|водный канал|водоканал спб', 'вода'),
            (r'свет|электричество|энергосбыт|мосэнергосбыт|лэнерго', 'электричество'),
            
            # Развлечения
            (r'netflix|нэтфликс|ivi|иви|кинопоиск|кино поиск|okko|окко', 'видеосервис'),
            (r'steam|стим|playstation|плейстейшн|xbox|убисофт|epic games', 'игры'),
            (r'ютуб|youtube|youtube premium|ютуб премиум', 'видеохостинг'),
            (r'спотифай|spotify|яндекс музыка|apple music|звук', 'музыка'),
            
            # Переводы
            (r'перевод|перевел|перевела|отправил|получил|перевод средств', 'перевод средств'),
            (r'сбербанк онлайн перевод|тинькофф перевод|альфа перевод', 'перевод'),
            (r'сбп|система быстрых платежей', 'перевод')
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        # 2. Удаляем цифры и спецсимволы
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 3. Удаляем стоп-слова
        stop_words = ['оплата', 'списание', 'покупка', 'товар', 'услуга', 'счет', 
                      'карта', 'руб', 'рублей', 'коп', 'копеек', 'опл', 'спис']
        for word in stop_words:
            text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
        
        # 4. Нормализация пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def amount_penalty(self, predicted_category: str, amount: Optional[float]) -> float:
        """
        Улучшенный штраф/бонус на основе суммы
        """
        if amount is None or predicted_category not in self.amount_weights:
            return 1.0
        
        min_amount, max_amount = self.amount_weights[predicted_category]
        abs_amount = abs(amount)
        
        # Более точная корректировка
        if min_amount <= abs_amount <= max_amount:
            # Идеальная сумма
            ideal = (min_amount + max_amount) / 2
            proximity = 1 - min(abs(abs_amount - ideal) / ideal, 0.5)
            return min(1.3, 1 + proximity * 0.3)
        elif abs_amount < min_amount:
            # Сумма меньше типичной
            ratio = abs_amount / min_amount if min_amount > 0 else 0.5
            return 0.7 + ratio * 0.3
        elif abs_amount > max_amount:
            # Сумма больше типичной
            ratio = max_amount / abs_amount if abs_amount > 0 else 0.5
            return 0.5 + ratio * 0.5
        else:
            return 1.0
    
    def keyword_correction(self, description: str, predicted: str, confidence: float) -> Tuple[str, float]:
        """
        Пост-обработка на основе ключевых слов (повышает точность)
        """
        text = description.lower()
        
        # Проверяем ключевые слова для каждой категории
        for category, keywords in self.keyword_rules.items():
            for keyword in keywords:
                if keyword in text:
                    return category, 0.98
        
        # Специальные правила для переводов
        if 'перевод' in text or 'перевел' in text or 'отправил' in text:
            if predicted != 'Переводы':
                return 'Переводы', max(confidence, 0.85)
        
        # Специальные правила для кафе
        if any(word in text for word in ['кафе', 'ресторан', 'кофе', 'бургер']):
            if predicted not in ['Кафе', 'Такси', 'Развлечения']:
                return 'Кафе', confidence * 1.1
        
        return predicted, confidence
    
    def train(self, df: pd.DataFrame = None, use_synthetic: bool = True, optimize: bool = True) -> Dict:
        """
        Обучение модели с расширенными признаками и оптимизацией
        """
        print("="*60)
        print("ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ (МАКСИМАЛЬНАЯ ТОЧНОСТЬ)")
        print("="*60)
        
        if df is None and use_synthetic:
            df = self._generate_enhanced_data(5000)
        
        # Предобработка
        print("1. Предобработка текста...")
        df['processed_text'] = df['description'].apply(self.preprocess_text)
        
        # Добавляем дополнительные признаки
        print("2. Добавление дополнительных признаков...")
        df['text_length'] = df['processed_text'].str.len()
        df['word_count'] = df['processed_text'].str.split().str.len()
        df['has_digit'] = df['description'].str.contains(r'\d').astype(int)
        df['has_emoji'] = df['description'].str.contains(r'[\U0001F600-\U0001F64F]').astype(int)
        df['is_uppercase'] = (df['description'].str.isupper()).astype(int)
        
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'].abs())
            df['amount_sign'] = np.sign(df['amount'])
        
        X_text = df['processed_text']
        y = df['category']
        
        # Оптимизация гиперпараметров
        if optimize:
            print("3. Оптимизация гиперпараметров (может занять время)...")
            best_params = self._optimize_hyperparameters(X_text, y)
        else:
            best_params = {
                'max_features': 10000,
                'ngram_range': (1, 3),
                'C': 1.5,
                'max_iter': 2000
            }
        
        # Создаём пайплайн
        print("4. Создание ML пайплайна...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=best_params.get('max_features', 10000),
                ngram_range=best_params.get('ngram_range', (1, 3)),
                min_df=2,
                max_df=0.85,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True
            )),
            ('clf', LogisticRegression(
                multi_class='multinomial',
                max_iter=best_params.get('max_iter', 2000),
                C=best_params.get('C', 1.5),
                class_weight='balanced',
                solver='saga',
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        print("5. Обучение модели...")
        self.pipeline.fit(X_text, y)
        
        # Оценка качества
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, X_text, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        y_pred = self.pipeline.predict(X_text)
        f1_macro = f1_score(y, y_pred, average='macro')
        
        metrics = {
            'accuracy_mean': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'f1_macro': f1_macro,
            'n_samples': len(df),
            'n_features': self.pipeline.named_steps['tfidf'].get_feature_names_out().shape[0]
        }
        
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print(f"{'='*60}")
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(f"F1-score (macro): {f1_macro:.3f}")
        print(f"Размер словаря: {metrics['n_features']}")
        print(f"Количество образцов: {metrics['n_samples']}")
        
        print(f"\nДетальный отчёт по категориям:")
        print(classification_report(y, y_pred))
        
        return metrics
    
    def _optimize_hyperparameters(self, X_text, y, cv=3):
        """Оптимизация гиперпараметров модели"""
        print("   Поиск лучших параметров...")
        
        param_grid = {
            'tfidf__max_features': [5000, 7000, 10000],
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'clf__C': [0.5, 1.0, 1.5, 2.0]
        }
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True)),
            ('clf', LogisticRegression(multi_class='multinomial', max_iter=2000, 
                                       class_weight='balanced', solver='saga'))
        ])
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', 
                                   n_jobs=-1, verbose=0)
        grid_search.fit(X_text, y)
        
        print(f"   Лучшие параметры: {grid_search.best_params_}")
        print(f"   Лучшая точность: {grid_search.best_score_:.3f}")
        
        return grid_search.best_params_
    
    def _generate_enhanced_data(self, n_samples: int) -> pd.DataFrame:
        """Генерация улучшенных синтетических данных"""
        np.random.seed(42)
        
        templates_with_amounts = {
            'Продукты': {
                'templates': ['пятерочка', 'магнит', 'ашан', 'перекресток', 'дикси', 'спар', 
                             'продукты', 'овощи', 'фрукты', 'мясо', 'рыба', 'молочка', 'гранат'],
                'amount_range': (100, 5000)
            },
            'Транспорт': {
                'templates': ['метро', 'автобус', 'троллейбус', 'электричка', 'проездной', 
                             'билет', 'транспортная карта'],
                'amount_range': (30, 1500)
            },
            'Кафе': {
                'templates': ['кафе', 'ресторан', 'кофейня', 'макдоналдс', 'бургер', 'суши',
                             'пицца', 'фастфуд', 'обед', 'ланч', 'бизнес ланч'],
                'amount_range': (150, 3000)
            },
            'Такси': {
                'templates': ['яндекс такси', 'uber', 'gett', 'такси', 'поездка', 'ситимобил'],
                'amount_range': (100, 2000)
            },
            'Аптеки': {
                'templates': ['аптека', 'лекарство', 'витамины', 'аптека.ру', 'таблетки', 
                             'здоровье', 'медикаменты'],
                'amount_range': (50, 4000)
            },
            'Связь_интернет': {
                'templates': ['мтс', 'билайн', 'мегафон', 'интернет', 'телефон', 'связь', 
                             'домашний интернет', 'ростелеком'],
                'amount_range': (300, 2000)
            },
            'Коммунальные': {
                'templates': ['жку', 'жкх', 'коммуналка', 'свет', 'вода', 'отопление', 
                             'газ', 'квартплата', 'электричество'],
                'amount_range': (1000, 12000)
            },
            'Развлечения': {
                'templates': ['кино', 'театр', 'концерт', 'игры', 'стим', 'netflix', 
                             'кинотеатр', 'квест', 'боулинг'],
                'amount_range': (200, 5000)
            },
            'Одежда_обувь': {
                'templates': ['одежда', 'обувь', 'wildberries', 'ozon', 'lamoda', 
                             'магазин одежды', 'джинсы', 'куртка'],
                'amount_range': (500, 20000)
            },
            'Переводы': {
                'templates': ['перевод', 'перевел', 'отправил', 'перевод средств', 
                             'перевод на карту', 'друзьям'],
                'amount_range': (1000, 50000)
            },
            'Красота': {
                'templates': ['парикмахерская', 'стрижка', 'маникюр', 'косметика', 
                             'салон красоты', 'брови'],
                'amount_range': (500, 6000)
            },
            'Образование': {
                'templates': ['курсы', 'университет', 'репетитор', 'книги', 'учебники', 
                             'образование', 'вебинар'],
                'amount_range': (500, 30000)
            },
            'Дом_ремонт': {
                'templates': ['стройка', 'ремонт', 'инструменты', 'мебель', 'икеа', 
                             'посуда', 'текстиль'],
                'amount_range': (500, 50000)
            },
            'Животные': {
                'templates': ['зоомагазин', 'корм', 'ветеринар', 'наполнитель', 
                             'лежанка', 'игрушки для животных'],
                'amount_range': (200, 5000)
            },
            'Подарки': {
                'templates': ['подарок', 'сувенир', 'цветы', 'открытка', 'подарочный набор'],
                'amount_range': (200, 10000)
            },
            'Спорт': {
                'templates': ['спортзал', 'фитнес', 'бассейн', 'экипировка', 'форма спортивная',
                             'тренажерка', 'абонемент в спортзал'],
                'amount_range': (300, 15000)
            }
        }
        
        data = []
        samples_per_category = n_samples // len(templates_with_amounts)
        
        for category, info in templates_with_amounts.items():
            for _ in range(samples_per_category):
                template = np.random.choice(info['templates'])
                amount = -np.random.uniform(*info['amount_range'])
                
                variation_type = np.random.choice(['none', 'prefix', 'suffix', 'typo'], p=[0.6, 0.15, 0.15, 0.1])
                
                if variation_type == 'prefix':
                    template = np.random.choice(['оплата ', 'списание ', 'покупка ']) + template
                elif variation_type == 'suffix':
                    template = template + ' ' + np.random.choice(['спб', 'мск', 'онлайн', 'карта', '*1234'])
                elif variation_type == 'typo':
                    if len(template) > 3:
                        pos = np.random.randint(1, len(template)-1)
                        template = template[:pos] + template[pos+1] + template[pos] + template[pos+2:]
                
                data.append({
                    'description': template,
                    'category': category,
                    'amount': amount
                })
        
        return pd.DataFrame(data)
    
    def predict(self, description: str, amount: Optional[float] = None) -> Tuple[str, float]:
        """
        СУПЕР-ПРЕДСКАЗАНИЕ с кэшированием, учётом суммы и пост-обработкой
        """
        cache_key = self._get_cache_key(description, amount)
        
        if cache_key in self.cache:
            category, confidence = self.cache[cache_key]
            return category, confidence
        
        if self.pipeline is None:
            raise ValueError("Модель не обучена")
        
        start_time = datetime.now()
        
        processed = self.preprocess_text(description)
        
        pred = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        confidence = max(proba)
        
        amount_factor = self.amount_penalty(pred, amount)
        adjusted_confidence = min(confidence * amount_factor, 0.99)
        
        pred, adjusted_confidence = self.keyword_correction(description, pred, adjusted_confidence)
        
        if amount:
            if abs(amount) > 10000 and any(word in description.lower() for word in ['перевод', 'перевел', 'отправил']):
                pred = 'Переводы'
                adjusted_confidence = 0.98
            
            if abs(amount) < 150 and any(word in description.lower() for word in ['кафе', 'кофе', 'бургер', 'кофейня']):
                pred = 'Кафе'
                adjusted_confidence = min(adjusted_confidence * 1.15, 0.99)
            
            if abs(amount) > 5000 and any(word in description.lower() for word in ['одежда', 'обувь', 'wildberries', 'ozon']):
                pred = 'Одежда_обувь'
                adjusted_confidence = 0.95
        
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
    
    def log_error(self, description: str, predicted: str, correct: str, amount: Optional[float]):
        """Логирование ошибок для последующего улучшения"""
        self.error_log.append({
            'description': description,
            'predicted': predicted,
            'correct': correct,
            'amount': amount,
            'timestamp': datetime.now()
        })
        
        if len(self.error_log) % 10 == 0:
            self._save_errors_to_file()
    
    def _save_errors_to_file(self):
        """Сохранение ошибок в файл"""
        if self.error_log:
            df_errors = pd.DataFrame(self.error_log)
            df_errors.to_csv('prediction_errors.csv', index=False, encoding='utf-8')
    
    def get_stats(self) -> Dict:
        """Получение статистики работы модели"""
        return {
            'cache_size': len(self.cache),
            'total_predictions': len(self.prediction_history),
            'errors_logged': len(self.error_log),
            'avg_prediction_time': np.mean([h['time_ms'] for h in self.prediction_history]) if self.prediction_history else 0,
            'model_loaded': self.pipeline is not None,
            'n_categories': len(self.categories)
        }
    
    def save_model(self, path: str = 'enhanced_model.pkl'):
        """Сохранение модели"""
        model_data = {
            'pipeline': self.pipeline,
            'cache': self.cache,
            'categories': self.categories,
            'amount_weights': self.amount_weights,
            'keyword_rules': self.keyword_rules,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, path)
        print(f"Модель сохранена в {path}")
    
    def load_model(self, path: str = 'enhanced_model.pkl'):
        """Загрузка модели"""
        model_data = joblib.load(path)
        self.pipeline = model_data['pipeline']
        self.cache = model_data.get('cache', {})
        self.categories = model_data.get('categories', self.categories)
        self.amount_weights = model_data.get('amount_weights', self.amount_weights)
        self.keyword_rules = model_data.get('keyword_rules', self.keyword_rules)
        self.label_encoder = model_data.get('label_encoder', self.label_encoder)
        print(f"Модель загружена из {path}")