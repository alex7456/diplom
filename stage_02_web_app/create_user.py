# create_user.py
from database import SessionLocal, User, create_tables
import hashlib
import secrets

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((password + salt).encode())
    return f"{salt}:{hash_obj.hexdigest()}"

def create_test_user():
    create_tables()
    db = SessionLocal()
    
    # Проверяем существующих пользователей
    users = db.query(User).all()
    print(f"Существующие пользователи: {len(users)}")
    for u in users:
        print(f"  - {u.username} ({u.email})")
    
    # Создаём тестового пользователя если нет
    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        admin = User(
            email="admin@example.com",
            username="admin",
            hashed_password=hash_password("admin")
        )
        db.add(admin)
        db.commit()
        print("\n✅ Создан пользователь: admin / admin")
    else:
        print(f"\n✅ Пользователь admin уже существует")
    
    # Создаём второго тестового пользователя
    test = db.query(User).filter(User.username == "test").first()
    if not test:
        test = User(
            email="test@example.com",
            username="test",
            hashed_password=hash_password("test123")
        )
        db.add(test)
        db.commit()
        print("✅ Создан пользователь: test / test123")
    
    db.close()

if __name__ == "__main__":
    create_test_user()