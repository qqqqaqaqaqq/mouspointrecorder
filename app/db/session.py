from app.db.connection import get_connection
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.core.settings import settings

# DB 관련
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def init_db():
    with get_connection(database="postgres") as conn:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{settings.DB_NAME}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {settings.DB_NAME}")
                print(f"Database '{settings.DB_NAME}' created.")

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mouse_points (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL
            )
            ''')
        with conn.cursor() as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS macro_mouse_points (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL
            )
            ''')            
        conn.commit()

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)