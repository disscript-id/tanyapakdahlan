import sqlite3
import os

def init_db():
    # Mengarahkan file db ke folder 'data'
    db_path = os.path.join('data', 'pak_dahlan_ai.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tabel User
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            name TEXT,
            location_ip TEXT,
            location_detected TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tabel Log Percakapan
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            user_query TEXT,
            ai_response TEXT,
            intent_category TEXT, 
            sentiment_score TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database Pak Dahlan Siap di: {db_path}")

if __name__ == "__main__":
    init_db()