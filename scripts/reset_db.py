import sqlite3

conn = sqlite3.connect("data/pak_dahlan_ai.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM user_sessions")
cursor.execute("DELETE FROM chat_logs")
cursor.execute("DELETE FROM user_feedback")
cursor.execute("DELETE FROM access_requests")

conn.commit()
conn.close()

print("✅ Database berhasil dibersihkan (level 2)")