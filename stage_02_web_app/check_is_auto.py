# check_is_auto.py
from database import SessionLocal, Transaction

db = SessionLocal()

transactions = db.query(Transaction).all()

manual = [t for t in transactions if t.is_auto == True]
bank = [t for t in transactions if t.is_auto == False]
unknown = [t for t in transactions if t.is_auto is None]

print(f"Ручной ввод (is_auto=True): {len(manual)}")
print(f"Выписки (is_auto=False): {len(bank)}")
print(f"Не определено (is_auto=None): {len(unknown)}")

for t in transactions[:5]:
    print(f"  {t.description} | is_auto={t.is_auto}")

db.close()