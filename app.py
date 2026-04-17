import os
import sqlite3
import random
import time
import uuid
import traceback
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect, render_template, url_for
from werkzeug.security import generate_password_hash, check_password_hash

from dis_script_v1 import generate_answer


def generate_pin(length=6):
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "ganti-secret-key-anda")
app.permanent_session_lifetime = timedelta(minutes=60)

DB_PATH = "/data/pak_dahlan_ai.db"



MAX_ACTIVE_USERS = int(os.getenv("MAX_ACTIVE_USERS", "25"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "2026pro")
ADMIN_INSIGHT_PASSWORD = os.getenv("ADMIN_INSIGHT_PASSWORD", "insight2026")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://alamat-anda")


def is_mobile_request(req) -> bool:
    user_agent = (req.headers.get("User-Agent") or "").lower()

    mobile_keywords = [
        "android",
        "iphone",
        "ipod",
        "mobile",
        "blackberry",
        "windows phone",
        "opera mini",
        "iemobile",
    ]

    return any(keyword in user_agent for keyword in mobile_keywords)


def block_non_mobile_page():
    return render_template("mobile_only.html"), 403


def block_non_mobile_redirect():
    session["flash_error"] = "Fitur ini hanya bisa dibuka melalui HP."
    return redirect("/akses")


def block_non_mobile_json():
    return jsonify({
        "message": "Fitur ini hanya bisa dibuka melalui HP."
    }), 403


def require_mobile_page():
    if not is_mobile_request(request):
        return block_non_mobile_page()
    return None


def require_mobile_redirect():
    if not is_mobile_request(request):
        return block_non_mobile_redirect()
    return None


def require_mobile_json():
    if not is_mobile_request(request):
        return block_non_mobile_json()
    return None


# =========================
# DATABASE
# =========================
def get_db_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn, table_name, column_name, column_def):
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {col[1] for col in columns}
    if column_name not in existing:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")


def init_db():
    conn = get_db_connection()

    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            note TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login_at TIMESTAMP
        )
        '''
    )

    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_phone TEXT,
            user_name TEXT,
            feedback_type TEXT,
            message TEXT,
            related_question TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS access_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            note TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_phone TEXT,
            user_name TEXT,
            user_query TEXT,
            ai_response TEXT,
            intent_category TEXT,
            sentiment_score TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_phone TEXT,
            session_start TIMESTAMP,
            session_end TIMESTAMP,
            duration_seconds INTEGER,
            message_count INTEGER DEFAULT 0,
            ip_address TEXT
        )
        '''
    )

    ensure_column(conn, "users", "note", "note TEXT")
    ensure_column(conn, "users", "last_login_at", "last_login_at TIMESTAMP")
    ensure_column(conn, "access_requests", "status", "status TEXT DEFAULT 'pending'")
    ensure_column(conn, "chat_logs", "ip_address", "ip_address TEXT")
    ensure_column(conn, "chat_logs", "session_id", "session_id TEXT")
    ensure_column(conn, "chat_logs", "question_length", "question_length INTEGER")
    ensure_column(conn, "chat_logs", "answer_length", "answer_length INTEGER")
    ensure_column(conn, "chat_logs", "response_time_ms", "response_time_ms INTEGER")
    ensure_column(conn, "chat_logs", "is_fallback", "is_fallback INTEGER DEFAULT 0")
    ensure_column(conn, "chat_logs", "answer_quality", "answer_quality TEXT")
    ensure_column(conn, "chat_logs", "quality_note", "quality_note TEXT")
    ensure_column(conn, "chat_logs", "failure_reason", "failure_reason TEXT")
    ensure_column(conn, "chat_logs", "root_cause_label", "root_cause_label TEXT")
    ensure_column(conn, "user_feedback", "feedback_type", "feedback_type TEXT")
    ensure_column(conn, "user_feedback", "message", "message TEXT")
    ensure_column(conn, "user_feedback", "related_question", "related_question TEXT")
    conn.commit()
    conn.close()


def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    phone = phone.strip().replace(" ", "").replace("-", "")
    if phone.startswith("+62"):
        phone = "0" + phone[3:]
    return phone


def count_active_users():
    conn = get_db_connection()
    row = conn.execute("SELECT COUNT(*) AS total FROM users WHERE is_active = 1").fetchone()
    conn.close()
    return row["total"] if row else 0


def get_user_by_phone(phone: str):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
    conn.close()
    return user


def create_user(phone: str, password: str, name: str = "", note: str = "", is_active: int = 1):
    conn = get_db_connection()
    conn.execute(
        '''
        INSERT INTO users (phone, password_hash, name, note, is_active)
        VALUES (?, ?, ?, ?, ?)
        ''',
        (
            phone,
            generate_password_hash(password),
            name.strip(),
            note.strip(),
            is_active,
        ),
    )
    conn.commit()
    conn.close()


def seed_initial_users():
    raw = os.getenv("TEST_USERS", "").strip()
    if not raw:
        return

    items = [item.strip() for item in raw.split("|") if item.strip()]
    for item in items:
        parts = item.split(":")
        if len(parts) < 2:
            continue

        phone = normalize_phone(parts[0])
        password = parts[1].strip()
        name = parts[2].strip() if len(parts) >= 3 else ""

        if not phone or not password:
            continue

        existing = get_user_by_phone(phone)
        if existing:
            continue

        if count_active_users() >= MAX_ACTIVE_USERS:
            break

        create_user(phone=phone, password=password, name=name, note="seed user", is_active=1)


# =========================
# ADMIN AUTH
# =========================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            session.pop("admin_insight_unlocked", None)
            return redirect(url_for("admin_page"))

        session["admin_error"] = "Password admin tidak sesuai."
        return redirect(url_for("admin_login"))

    admin_error = session.pop("admin_error", None) if "admin_error" in session else None
    return render_template("admin_login.html", admin_error=admin_error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_insight_unlocked", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/insight-login", methods=["GET", "POST"])
def admin_insight_login():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        password = (request.form.get("password") or "").strip()

        if password == ADMIN_INSIGHT_PASSWORD:
            session["admin_insight_unlocked"] = True
            return redirect(url_for("admin_insight"))

        session["admin_error"] = "Password insight tidak sesuai."
        return redirect(url_for("admin_insight_login"))

    admin_error = session.pop("admin_error", None) if "admin_error" in session else None
    return render_template("admin_insight_login.html", admin_error=admin_error)


# =========================
# ADMIN PANEL
# =========================
@app.route("/admin")
def admin_page():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()

    requests = conn.execute(
        '''
        SELECT * FROM access_requests
        WHERE status = 'pending'
        ORDER BY created_at DESC
        '''
    ).fetchall()

    active_users = conn.execute(
        '''
        SELECT id, phone, name, note, created_at, last_login_at
        FROM users
        WHERE is_active = 1
        ORDER BY created_at DESC
        '''
    ).fetchall()

    conn.close()

    return render_template(
        "admin.html",
        requests=requests,
        active_users=active_users,
        active_user_count=count_active_users(),
        max_active_users=MAX_ACTIVE_USERS,
        insight_unlocked=session.get("admin_insight_unlocked", False),
    )


@app.route("/admin/approve/<int:req_id>")
def approve_request(req_id):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()

    req = conn.execute(
        '''
        SELECT * FROM access_requests WHERE id = ?
        ''',
        (req_id,),
    ).fetchone()

    if not req:
        conn.close()
        return "Request tidak ditemukan"

    if count_active_users() >= MAX_ACTIVE_USERS:
        conn.close()
        return "Kuota peserta sudah penuh"

    phone = normalize_phone(req["phone"])
    name = req["name"]

    existing = conn.execute(
        "SELECT id FROM users WHERE phone = ?",
        (phone,),
    ).fetchone()

    if existing:
        conn.execute(
            '''
            UPDATE access_requests
            SET status = 'approved'
            WHERE id = ?
            ''',
            (req_id,),
        )
        conn.commit()
        conn.close()
        return "Nomor ini sudah pernah diaktifkan."

    pin = generate_pin()

    conn.execute(
        '''
        INSERT INTO users (phone, password_hash, name, note, is_active)
        VALUES (?, ?, ?, ?, 1)
        ''',
        (
            phone,
            generate_password_hash(pin),
            name,
            req["note"] if "note" in req.keys() else "",
        ),
    )

    conn.execute(
        '''
        UPDATE access_requests
        SET status = 'approved'
        WHERE id = ?
        ''',
        (req_id,),
    )

    conn.commit()
    conn.close()

    return render_template(
        "admin_approved.html",
        name=name,
        phone=phone,
        pin=pin,
        app_base_url=APP_BASE_URL,
    )


@app.route("/admin/insight")
def admin_insight():
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    if not session.get("admin_insight_unlocked"):
        return redirect("/admin/insight-login")

    conn = get_db_connection()

    total_users = conn.execute("""
        SELECT COUNT(*) FROM users WHERE is_active = 1
    """).fetchone()[0]

    total_questions = conn.execute("""
        SELECT COUNT(*) FROM chat_logs
    """).fetchone()[0]

    total_unique_chat_users = conn.execute("""
        SELECT COUNT(DISTINCT user_phone)
        FROM chat_logs
        WHERE user_phone IS NOT NULL AND TRIM(user_phone) != ''
    """).fetchone()[0]

    total_sessions = conn.execute("""
        SELECT COUNT(*) FROM user_sessions
    """).fetchone()[0]

    avg_response = conn.execute("""
        SELECT AVG(response_time_ms) FROM chat_logs
    """).fetchone()[0]

    avg_question_length = conn.execute("""
        SELECT AVG(question_length) FROM chat_logs
    """).fetchone()[0]

    avg_answer_length = conn.execute("""
        SELECT AVG(answer_length) FROM chat_logs
    """).fetchone()[0]

    avg_questions_per_user = conn.execute("""
        SELECT AVG(user_total) FROM (
            SELECT COUNT(*) AS user_total
            FROM chat_logs
            WHERE user_phone IS NOT NULL AND TRIM(user_phone) != ''
            GROUP BY user_phone
        )
    """).fetchone()[0]

    avg_session_duration = conn.execute("""
        SELECT AVG(duration_seconds) FROM user_sessions
        WHERE duration_seconds IS NOT NULL
    """).fetchone()[0]

    per_day = conn.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as total
        FROM chat_logs
        GROUP BY day
        ORDER BY day
    """).fetchall()

    per_hour = conn.execute("""
        SELECT strftime('%H', created_at) as hour, COUNT(*) as total
        FROM chat_logs
        GROUP BY hour
        ORDER BY hour
    """).fetchall()

    sessions_per_day = conn.execute("""
        SELECT DATE(session_start) as day, COUNT(*) as total
        FROM user_sessions
        WHERE session_start IS NOT NULL
        GROUP BY DATE(session_start)
        ORDER BY day
    """).fetchall()

    category_data = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(intent_category), ''), 'Umum') as intent_category,
            COUNT(*) as total
        FROM chat_logs
        GROUP BY COALESCE(NULLIF(TRIM(intent_category), ''), 'Umum')
        ORDER BY total DESC
    """).fetchall()

    top_questions = conn.execute("""
        SELECT
            user_query,
            COUNT(*) as total
        FROM chat_logs
        WHERE user_query IS NOT NULL AND TRIM(user_query) != ''
        GROUP BY user_query
        ORDER BY total DESC, user_query ASC
        LIMIT 10
    """).fetchall()

    top_users = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            COUNT(*) as total
        FROM chat_logs
        GROUP BY COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama')
        ORDER BY total DESC, user_name ASC
        LIMIT 10
    """).fetchall()

    longest_questions = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            user_query,
            question_length,
            created_at
        FROM chat_logs
        WHERE user_query IS NOT NULL AND TRIM(user_query) != ''
        ORDER BY question_length DESC, created_at DESC
        LIMIT 10
    """).fetchall()

    slowest_responses = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            user_query,
            response_time_ms,
            created_at
        FROM chat_logs
        WHERE response_time_ms IS NOT NULL
        ORDER BY response_time_ms DESC, created_at DESC
        LIMIT 10
    """).fetchall()

    shortest_answers = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            user_query,
            ai_response,
            answer_length,
            created_at
        FROM chat_logs
        WHERE ai_response IS NOT NULL AND TRIM(ai_response) != ''
        ORDER BY answer_length ASC, created_at DESC
        LIMIT 10
    """).fetchall()

    quality_data = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(answer_quality), ''), 'unknown') as answer_quality,
            COUNT(*) as total
        FROM chat_logs
        GROUP BY COALESCE(NULLIF(TRIM(answer_quality), ''), 'unknown')
        ORDER BY total DESC
    """).fetchall()

    total_fallback = conn.execute("""
        SELECT COUNT(*) FROM chat_logs
        WHERE is_fallback = 1
    """).fetchone()[0]

    total_short = conn.execute("""
        SELECT COUNT(*) FROM chat_logs
        WHERE answer_quality = 'short'
    """).fetchone()[0]

    total_weak = conn.execute("""
        SELECT COUNT(*) FROM chat_logs
        WHERE answer_quality = 'weak'
    """).fetchone()[0]

    total_good = conn.execute("""
        SELECT COUNT(*) FROM chat_logs
        WHERE answer_quality = 'good'
    """).fetchone()[0]

    total_scored = total_good + total_short + total_weak + total_fallback

    if total_scored > 0:
        satisfaction = (
            (total_good * 1.0) +
            (total_short * 0.5) +
            (total_weak * 0.2) +
            (total_fallback * 0.0)
        ) / total_scored * 100
    else:
        satisfaction = 0

    satisfaction = round(satisfaction, 1)

    fallback_by_category = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(intent_category), ''), 'Umum') as intent_category,
            COUNT(*) as total
        FROM chat_logs
        WHERE is_fallback = 1
        GROUP BY COALESCE(NULLIF(TRIM(intent_category), ''), 'Umum')
        ORDER BY total DESC
        LIMIT 10
    """).fetchall()

    fallback_logs = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            user_query,
            ai_response,
            quality_note,
            created_at
        FROM chat_logs
        WHERE is_fallback = 1
        ORDER BY created_at DESC
        LIMIT 10
    """).fetchall()

    root_cause_data = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(root_cause_label), ''), 'unknown') as root_cause_label,
            COUNT(*) as total
        FROM chat_logs
        GROUP BY COALESCE(NULLIF(TRIM(root_cause_label), ''), 'unknown')
        ORDER BY total DESC
    """).fetchall()

    top_failure_reasons = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(failure_reason), ''), 'Tidak diketahui') as failure_reason,
            COUNT(*) as total
        FROM chat_logs
        WHERE root_cause_label IS NOT NULL
          AND root_cause_label != 'ok'
        GROUP BY COALESCE(NULLIF(TRIM(failure_reason), ''), 'Tidak diketahui')
        ORDER BY total DESC
        LIMIT 10
    """).fetchall()

    root_cause_logs = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            user_query,
            answer_quality,
            root_cause_label,
            failure_reason,
            created_at
        FROM chat_logs
        WHERE root_cause_label IS NOT NULL
          AND root_cause_label != 'ok'
        ORDER BY created_at DESC
        LIMIT 10
    """).fetchall()

    total_feedback = conn.execute("""
        SELECT COUNT(*) FROM user_feedback
    """).fetchone()[0]

    feedback_by_type = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(feedback_type), ''), 'umum') as feedback_type,
            COUNT(*) as total
        FROM user_feedback
        GROUP BY COALESCE(NULLIF(TRIM(feedback_type), ''), 'umum')
        ORDER BY total DESC
    """).fetchall()

    latest_feedback = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(user_name), ''), user_phone, 'Tanpa Nama') as user_name,
            feedback_type,
            message,
            related_question,
            created_at
        FROM user_feedback
        ORDER BY created_at DESC
        LIMIT 10
    """).fetchall()

    mood_data = conn.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(sentiment_score), ''), 'Netral') as mood,
            COUNT(*) as total
        FROM chat_logs
        GROUP BY mood
        ORDER BY total DESC
    """).fetchall()

    conn.close()

    print("QUALITY DATA:", quality_data)
    print("ROOT CAUSE DATA:", root_cause_data)

    return render_template(
        "admin_insight.html",
        total_users=total_users,
        total_questions=total_questions,
        total_unique_chat_users=total_unique_chat_users,
        total_sessions=total_sessions,
        avg_response=int(avg_response or 0),
        avg_question_length=round(avg_question_length or 0, 1),
        avg_answer_length=round(avg_answer_length or 0, 1),
        avg_questions_per_user=round(avg_questions_per_user or 0, 1),
        avg_session_duration=int(avg_session_duration or 0),
        per_day=per_day,
        per_hour=per_hour,
        sessions_per_day=sessions_per_day,
        category_data=category_data,
        top_questions=top_questions,
        top_users=top_users,
        longest_questions=longest_questions,
        slowest_responses=slowest_responses,
        shortest_answers=shortest_answers,
        quality_data=quality_data,
        total_fallback=total_fallback,
        total_short=total_short,
        total_weak=total_weak,
        total_good=total_good,
        fallback_by_category=fallback_by_category,
        fallback_logs=fallback_logs,
        root_cause_data=root_cause_data,
        top_failure_reasons=top_failure_reasons,
        root_cause_logs=root_cause_logs,
        total_feedback=total_feedback,
        feedback_by_type=feedback_by_type,
        latest_feedback=latest_feedback,
        satisfaction=satisfaction,
        mood_data=mood_data
    )


@app.route("/admin/reset/<phone>")
def reset_pin(phone):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()

    user = conn.execute(
        '''
        SELECT * FROM users WHERE phone = ?
        ''',
        (phone,),
    ).fetchone()

    if not user:
        conn.close()
        return "User tidak ditemukan"

    pin = generate_pin()

    conn.execute(
        '''
        UPDATE users
        SET password_hash = ?
        WHERE phone = ?
        ''',
        (
            generate_password_hash(pin),
            phone,
        ),
    )

    conn.commit()
    conn.close()

    return render_template(
        "admin_approved.html",
        name=user["name"] if user["name"] else phone,
        phone=phone,
        pin=pin,
        app_base_url=APP_BASE_URL,
    )


@app.route("/admin/deactivate/<int:user_id>", methods=["POST"])
def deactivate_user(user_id):
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET is_active = 0 WHERE id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()

    return redirect("/admin")


@app.route("/admin/activate/<int:user_id>", methods=["POST"])
def activate_user(user_id):
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    conn = get_db_connection()

    conn.execute(
        "UPDATE users SET is_active = 1 WHERE id = ?",
        (user_id,)
    )

    conn.commit()
    conn.close()

    return redirect("/admin")


# =========================
# PUBLIC ROUTES
# =========================
@app.route("/")
def home():
    blocked = require_mobile_page()
    if blocked:
        return blocked

    if session.get("logged_in"):
        return redirect(url_for("chat_page"))
    return render_template("landing_fake.html")


@app.route("/akses")
def akses_page():
    blocked = require_mobile_page()
    if blocked:
        return blocked

    logged_in = session.get("logged_in", False)
    user_name = session.get("user_name", "Tamu")
    active_user_count = count_active_users()

    return render_template(
        "index_dis.html",
        logged_in=logged_in,
        user_name=user_name,
        active_user_count=active_user_count,
        max_active_users=MAX_ACTIVE_USERS,
    )


@app.route("/chat")
def chat_page():
    blocked = require_mobile_redirect()
    if blocked:
        return blocked

    if not session.get("logged_in"):
        return redirect("/akses")

    user_name = session.get("user_name", "Tamu")
    active_user_count = count_active_users()

    return render_template(
        "index_dis.html",
        logged_in=True,
        user_name=user_name,
        active_user_count=active_user_count,
        max_active_users=MAX_ACTIVE_USERS,
    )


@app.route("/request-access", methods=["POST"])
def request_access():
    blocked = require_mobile_redirect()
    if blocked:
        return blocked

    name = (request.form.get("name") or "").strip()
    phone = normalize_phone(request.form.get("phone") or "")
    note = (request.form.get("note") or "").strip()

    if not name or not phone:
        return redirect("/akses")



    conn = get_db_connection()

    # ✅ CEK: apakah nomor sudah jadi user aktif
    existing_user = conn.execute(
        "SELECT id FROM users WHERE phone = ?",
        (phone,)
    ).fetchone()

    if existing_user:
        conn.close()
        session["flash_error"] = "Nomor WhatsApp ini sudah aktif. Silakan login."
        return redirect("/akses")

    # 🔴 CEK: apakah nomor sudah pernah daftar
    existing = conn.execute(
        "SELECT id, status FROM access_requests WHERE phone = ?",
        (phone,)
    ).fetchone()

    if existing:
        conn.close()

        session["flash_error"] = "Nomor WhatsApp ini sudah terdaftar dan masih dalam antrean persetujuan."
        return redirect("/akses")

    # 🟢 kalau belum ada → lanjut simpan
    conn.execute(
        '''
        INSERT INTO access_requests (name, phone, note, status)
        VALUES (?, ?, ?, 'pending')
        ''',
        (name, phone, note),
    )

    conn.commit()
    conn.close()

    session["flash_message"] = "Permintaan akses berhasil dikirim. Silakan tunggu persetujuan."
    return redirect("/akses")


@app.route("/login", methods=["POST"])
def login():
    blocked = require_mobile_redirect()
    if blocked:
        return blocked

    phone = normalize_phone(request.form.get("phone") or "")
    pin = (request.form.get("password") or "").strip()

    if not phone or not pin:
        session["flash_error"] = "Nomor WhatsApp dan PIN wajib diisi."
        return redirect("/akses")

    user = get_user_by_phone(phone)

    if not user:
        session["flash_error"] = "Nomor belum terdaftar atau belum diaktifkan."
        return redirect("/akses")

    if int(user["is_active"]) != 1:
        session["flash_error"] = "Akses Anda belum aktif."
        return redirect("/akses")

    if not check_password_hash(user["password_hash"], pin):
        session["flash_error"] = "PIN tidak sesuai."
        return redirect("/akses")

    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
        (user["id"],),
    )
    conn.commit()
    conn.close()

    session.permanent = True
    session["logged_in"] = True
    session["user_id"] = user["id"]
    session["user_phone"] = user["phone"]
    session["user_name"] = user["name"] if user["name"] else user["phone"]
    session["chat_history"] = []
    session["session_id"] = str(uuid.uuid4())
    session["session_start"] = time.time()

    return redirect("/chat")


@app.route("/logout")
def logout():

    blocked = require_mobile_redirect()
    if blocked:
        return blocked

    if session.get("session_id"):
        duration = int(time.time() - session.get("session_start", time.time()))

        conn = get_db_connection()
        conn.execute(
            '''
            INSERT INTO user_sessions (
                session_id, user_phone, session_start, session_end,
                duration_seconds, message_count, ip_address
            )
            VALUES (?, ?, datetime('now', ?), datetime('now'), ?, ?, ?)
            ''',
            (
                session.get("session_id"),
                session.get("user_phone"),
                f"-{duration} seconds",
                duration,
                len(session.get("chat_history", [])),
                request.remote_addr,
            ),
        )
        conn.commit()
        conn.close()

    session.pop("logged_in", None)
    session.pop("user_id", None)
    session.pop("user_phone", None)
    session.pop("user_name", None)
    session.pop("chat_history", None)
    session.pop("session_id", None)
    session.pop("session_start", None)

    return redirect("/")


def evaluate_answer_quality(question: str, answer: str, response_time_ms: int = 0):
    question = (question or "").strip()
    answer = (answer or "").strip()

    answer_length = len(answer)

    fallback_markers = [
        "saya belum menemukan rujukan yang cukup kuat",
        "saya belum menemukan rujukan",
        "rujukan belum cukup kuat",
        "coba pertanyaannya dipersempit",
        "coba ulangi dengan detail yang lebih spesifik",
        "maaf, sistem pencarian referensi sedang bermasalah",
        "maaf, layanan model sedang bermasalah",
        "maaf, terjadi gangguan internal",
        "maaf, respons model terlalu lama",
        "maaf, trafik sedang padat",
        "silakan login terlebih dahulu",
        "pertanyaannya masih kosong",
    ]

    lower_answer = answer.lower()

    is_fallback = 0
    quality = "good"
    note = "Jawaban normal."

    if not answer:
        is_fallback = 1
        quality = "fallback"
        note = "Jawaban kosong."
        return {
            "is_fallback": is_fallback,
            "answer_quality": quality,
            "quality_note": note,
        }

    if any(marker in lower_answer for marker in fallback_markers):
        is_fallback = 1
        quality = "fallback"
        note = "Jawaban terdeteksi sebagai fallback / gagal menjawab kuat."
        return {
            "is_fallback": is_fallback,
            "answer_quality": quality,
            "quality_note": note,
        }

    if answer_length < 80:
        quality = "short"
        note = "Jawaban terlalu pendek."
        return {
            "is_fallback": is_fallback,
            "answer_quality": quality,
            "quality_note": note,
        }

    weak_markers = [
        "mungkin",
        "barangkali",
        "bisa jadi",
        "kurang tahu",
        "tidak terlalu mendalami",
        "belum cukup kuat",
    ]

    weak_hits = sum(1 for marker in weak_markers if marker in lower_answer)

    if weak_hits >= 2:
        quality = "weak"
        note = "Jawaban ada, tetapi terindikasi lemah / kurang tegas."
        return {
            "is_fallback": is_fallback,
            "answer_quality": quality,
            "quality_note": note,
        }

    if response_time_ms and response_time_ms > 15000:
        quality = "weak"
        note = "Jawaban ada, tetapi response sangat lambat."
        return {
            "is_fallback": is_fallback,
            "answer_quality": quality,
            "quality_note": note,
        }

    return {
        "is_fallback": is_fallback,
        "answer_quality": quality,
        "quality_note": note,
    }


def detect_root_cause(question: str, answer: str, quality_result: dict):
    question = (question or "").strip()
    answer = (answer or "").strip().lower()

    is_fallback = int(quality_result.get("is_fallback", 0))
    answer_quality = (quality_result.get("answer_quality") or "").strip().lower()

    generic_markers = [
        "gimana",
        "bagaimana",
        "apa pendapat",
        "menurut anda",
        "jelaskan dong",
        "coba jelaskan",
    ]

    no_reference_markers = [
        "saya belum menemukan rujukan yang cukup kuat",
        "saya belum menemukan rujukan",
        "rujukan belum cukup kuat",
        "coba pertanyaannya dipersempit",
        "coba ulangi dengan detail yang lebih spesifik",
    ]

    system_issue_markers = [
        "maaf, sistem pencarian referensi sedang bermasalah",
        "maaf, layanan model sedang bermasalah",
        "maaf, terjadi gangguan internal",
        "maaf, respons model terlalu lama",
        "maaf, trafik sedang padat",
    ]

    login_or_input_markers = [
        "silakan login terlebih dahulu",
        "pertanyaannya masih kosong",
    ]

    weak_confidence_markers = [
        "mungkin",
        "barangkali",
        "bisa jadi",
        "kurang tahu",
        "tidak terlalu mendalami",
        "belum cukup kuat",
    ]

    if any(marker in answer for marker in login_or_input_markers):
        return {
            "failure_reason": "Masalah akses atau input user.",
            "root_cause_label": "login_or_input"
        }

    if any(marker in answer for marker in system_issue_markers):
        return {
            "failure_reason": "Gangguan sistem, model, atau timeout.",
            "root_cause_label": "system_issue"
        }

    if any(marker in answer for marker in no_reference_markers):
        return {
            "failure_reason": "Sistem tidak menemukan referensi yang cukup kuat.",
            "root_cause_label": "no_reference"
        }

    if is_fallback == 1 and len(question) < 25:
        return {
            "failure_reason": "Pertanyaan terlalu pendek atau terlalu umum.",
            "root_cause_label": "too_generic"
        }

    if is_fallback == 1 and any(marker in question.lower() for marker in generic_markers):
        return {
            "failure_reason": "Pertanyaan terlalu umum atau kurang spesifik.",
            "root_cause_label": "too_generic"
        }

    weak_hits = sum(1 for marker in weak_confidence_markers if marker in answer)
    if answer_quality == "weak" or weak_hits >= 2:
        return {
            "failure_reason": "Jawaban ada, tetapi kepercayaan / ketegasannya lemah.",
            "root_cause_label": "weak_confidence"
        }

    return {
        "failure_reason": "Tidak terdeteksi masalah utama.",
        "root_cause_label": "ok"
    }


def classify_feedback_type(message: str) -> str:
    text = (message or "").strip().lower()

    if not text:
        return "umum"

    bug_keywords = [
        "error", "bug", "tidak muncul", "gagal", "blank", "macet",
        "loading", "lemot", "force close", "crash"
    ]
    answer_keywords = [
        "jawaban", "tidak nyambung", "salah", "ngawur", "kurang tepat",
        "halusinasi", "tidak sesuai", "salah paham"
    ]
    ui_keywords = [
        "tampilan", "ui", "ux", "desain", "layout", "tombol",
        "warna", "font", "scroll", "halaman"
    ]
    suggestion_keywords = [
        "saran", "fitur", "tambahkan", "sebaiknya", "usul",
        "lebih bagus", "lebih baik", "mohon ditambah"
    ]

    if any(word in text for word in bug_keywords):
        return "bug"
    if any(word in text for word in answer_keywords):
        return "jawaban"
    if any(word in text for word in ui_keywords):
        return "ui_ux"
    if any(word in text for word in suggestion_keywords):
        return "saran"

    return "umum"


def classify_mood(user_query):
    text = (user_query or "").strip()

    if not text:
        return "Netral"

    prompt = f"""
Klasifikasikan mood dari kalimat berikut ke salah satu label:

Positif, Netral, Negatif

Aturan:
- Positif: antusias, senang, puas
- Netral: pertanyaan biasa tanpa emosi kuat
- Negatif: kecewa, marah, bingung, frustrasi

Jawab hanya 1 kata saja.

Kalimat:
{text}

Jawaban:
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_INPUTCORPUS"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Anda classifier mood yang sederhana dan disiplin."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        mood = response.choices[0].message.content.strip().capitalize()

        if mood not in ["Positif", "Netral", "Negatif"]:
            return "Netral"

        return mood

    except Exception as e:
        print("Mood classify error:", e)
        print(traceback.format_exc())
        return "Netral"


@app.route("/ask", methods=["POST"])
def ask():

    blocked = require_mobile_json()
    if blocked:
        return blocked

    if not session.get("logged_in"):
        return jsonify({"answer": "Silakan login terlebih dahulu."}), 401

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Pertanyaannya masih kosong."}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]
    recent_history = chat_history[-6:]

    time.sleep(0.5)

    start_time = time.time()
    answer = generate_answer(question, recent_history)

    end_time = time.time()
    response_time = int((end_time - start_time) * 1000)

    chat_history.append({"q": question, "a": answer})
    session["chat_history"] = chat_history[-12:]
    session.modified = True

    mood = classify_mood(question)

    try:
        from analisis import classify_intent
        category = classify_intent(question)
    except Exception:
        category = "Umum"

    quality_result = evaluate_answer_quality(
        question=question,
        answer=answer,
        response_time_ms=response_time
    )

    root_cause_result = detect_root_cause(
        question=question,
        answer=answer,
        quality_result=quality_result
    )

    try:
        conn = get_db_connection()
        conn.execute(
            '''
            INSERT INTO chat_logs (
                user_phone, user_name, user_query, ai_response,
                intent_category, sentiment_score, ip_address,
                session_id, question_length, answer_length, response_time_ms,
                is_fallback, answer_quality, quality_note,
                failure_reason, root_cause_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                session.get("user_phone", "Guest"),
                session.get("user_name", "Guest"),
                question,
                answer,
                category,
                mood,
                request.remote_addr,
                session.get("session_id"),
                len(question),
                len(answer),
                response_time,
                quality_result["is_fallback"],
                quality_result["answer_quality"],
                quality_result["quality_note"],
                root_cause_result["failure_reason"],
                root_cause_result["root_cause_label"],
            ),
        )
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Log Error: {e}")
        print(traceback.format_exc())

    return jsonify({"answer": answer})


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():

    blocked = require_mobile_json()
    if blocked:
        return blocked

    if not session.get("logged_in"):
        return jsonify({"status": "error", "message": "Silakan login terlebih dahulu."}), 401

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    related_question = (data.get("related_question") or "").strip()

    if not message:
        return jsonify({"status": "error", "message": "Masukan masih kosong."}), 400

    feedback_type = classify_feedback_type(message)

    try:
        conn = get_db_connection()
        conn.execute(
            '''
            INSERT INTO user_feedback (
                user_phone, user_name, feedback_type, message, related_question
            )
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                session.get("user_phone", "Guest"),
                session.get("user_name", "Guest"),
                feedback_type,
                message,
                related_question
            ),
        )
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Feedback Error: {e}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": "Gagal menyimpan masukan."}), 500

    return jsonify({"status": "ok", "message": "Masukan berhasil disimpan."})


@app.context_processor
def inject_flash_messages():
    flash_message = session.pop("flash_message", None) if "flash_message" in session else None
    flash_error = session.pop("flash_error", None) if "flash_error" in session else None
    admin_error = session.pop("admin_error", None) if "admin_error" in session else None
    return dict(
        flash_message=flash_message,
        flash_error=flash_error,
        admin_error=admin_error,
    )


init_db()
seed_initial_users()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)