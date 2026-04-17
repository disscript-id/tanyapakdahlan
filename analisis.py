import sqlite3
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_INPUTCORPUS"))

VALID_CATEGORIES = [
    "PERSONAL",
    "BISNIS",
    "POLITIK",
    "INTERNASIONAL",
    "TEKNOLOGI",
    "PERTANIAN",
    "KESEHATAN",
    "MOTIVASI",
    "CERITA",
    "SOSIAL",
    "LAINNYA"
]


def classify_intent(query):
    query_clean = (query or "").strip()
    query_lower = query_clean.lower()

    if not query_clean:
        return "LAINNYA"

    prompt = f"""
Anda adalah sistem klasifikasi topik tulisan Dahlan Iskan.

Tugas:
Klasifikasikan pertanyaan user ke dalam SALAH SATU kategori berikut:

PERSONAL
BISNIS
POLITIK
INTERNASIONAL
TEKNOLOGI
PERTANIAN
KESEHATAN
MOTIVASI
CERITA
SOSIAL
LAINNYA

Definisi kategori:
- PERSONAL: kehidupan pribadi, keluarga, pengalaman hidup, keseharian
- BISNIS: ekonomi, perusahaan, industri, BUMN, investasi, perdagangan
- POLITIK: politik dalam negeri, kekuasaan, pemerintahan, kebijakan nasional
- INTERNASIONAL: luar negeri, geopolitik, negara lain, kebijakan global
- TEKNOLOGI: inovasi, AI, mobil listrik, teknologi industri, teknologi umum
- PERTANIAN: pertanian, peternakan, pangan, produksi dasar
- KESEHATAN: medis, rumah sakit, dokter, pengalaman berobat, kesehatan tubuh
- MOTIVASI: inspirasi, pendidikan, nilai hidup, kerja keras, semangat sukses
- CERITA: kisah tokoh, profil orang, cerita menarik, storytelling
- SOSIAL: agama, budaya, masyarakat, komunitas, hubungan sosial
- LAINNYA: jika tidak jelas atau tidak cocok

Aturan:
- Jawab hanya dengan 1 kata kategori
- Jangan menambah penjelasan
- Jangan membuat kategori baru
- Jika ragu, pilih LAINNYA

Contoh:
"Pak Dahlan makan apa?" -> PERSONAL
"Bagaimana ekonomi Indonesia?" -> BISNIS
"Presiden mengambil kebijakan apa?" -> POLITIK
"China vs Amerika bagaimana?" -> INTERNASIONAL
"Mobil listrik bagus?" -> TEKNOLOGI
"Nasib petani bagaimana?" -> PERTANIAN
"Rumah sakit di Tiongkok seperti apa?" -> KESEHATAN
"Bagaimana cara sukses?" -> MOTIVASI
"Ceritakan tokoh inspiratif" -> CERITA
"Hubungan antaragama bagaimana?" -> SOSIAL

Pertanyaan:
{query_clean}

Jawaban:
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Anda adalah classifier topik yang sangat disiplin dan hanya boleh menjawab dengan satu kategori yang valid."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=20
        )

        category = (response.choices[0].message.content or "").strip().upper()

        if category in VALID_CATEGORIES:
            return category

        return "LAINNYA"

    except Exception as e:
        print("AI classify error:", e)

        # fallback keyword sederhana agar sistem tetap jalan
        keyword_patterns = {
            "TEKNOLOGI": [
                r"\bmobil\b", r"\blistrik\b", r"\btesla\b", r"\bteknologi\b",
                r"\bai\b", r"\bartificial intelligence\b", r"\baplikasi\b",
                r"\bapps\b", r"\bweb\b", r"\bscript\b", r"\bdigital\b"
            ],
            "KESEHATAN": [
                r"\bsehat\b", r"\bsakit\b", r"\bmakan\b", r"\bolahraga\b",
                r"\blari\b", r"\bdiet\b", r"\blambung\b", r"\bmaag\b",
                r"\bpencernaan\b", r"\brumah sakit\b", r"\bdokter\b",
                r"\bobat\b", r"\bmedis\b"
            ],
            "BISNIS": [
                r"\bbisnis\b", r"\bekonomi\b", r"\bsaham\b", r"\binvestasi\b",
                r"\bduit\b", r"\buang\b", r"\bkerja\b", r"\bkantor\b",
                r"\bperusahaan\b", r"\bbumn\b", r"\bdagang\b", r"\bindustri\b"
            ],
            "POLITIK": [
                r"\bpolitik\b", r"\bpemerintah\b", r"\bpresiden\b", r"\bmenteri\b",
                r"\bnegara\b", r"\brakyat\b", r"\bkebijakan\b", r"\bparlemen\b",
                r"\bdpr\b", r"\bpemilu\b"
            ],
            "INTERNASIONAL": [
                r"\bchina\b", r"\btiongkok\b", r"\bamerika\b", r"\bjepang\b",
                r"\beropa\b", r"\bglobal\b", r"\binternasional\b", r"\bluar negeri\b",
                r"\bisrael\b", r"\biran\b", r"\brusia\b", r"\bukraina\b"
            ],
            "PERTANIAN": [
                r"\bpetani\b", r"\bpertanian\b", r"\bsawah\b", r"\bpadi\b",
                r"\bpangan\b", r"\bternak\b", r"\bpeternakan\b", r"\bkebun\b"
            ],
            "MOTIVASI": [
                r"\bsukses\b", r"\bsemangat\b", r"\bmotivasi\b", r"\binspirasi\b",
                r"\bpendidikan\b", r"\bbelajar\b", r"\bkerja keras\b"
            ],
            "SOSIAL": [
                r"\bagama\b", r"\bmasyarakat\b", r"\bsosial\b", r"\bbudaya\b",
                r"\bkomunitas\b", r"\blintas agama\b"
            ],
            "CERITA": [
                r"\bcerita\b", r"\bkisah\b", r"\btokoh\b", r"\bprofil\b",
                r"\bsiapa dia\b", r"\borang ini\b"
            ],
            "PERSONAL": [
                r"\bistri\b", r"\banak\b", r"\bcucu\b", r"\bkeluarga\b",
                r"\bpribadi\b", r"\bpak dahlan\b", r"\bpengalaman\b",
                r"\bperjalanan\b"
            ],
        }

        for category, patterns in keyword_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return category

        return "LAINNYA"


def generate_full_report():
    db_path = os.path.join("data", "pak_dahlan_ai.db")
    if not os.path.exists(db_path):
        print("Database belum ditemukan.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, user_query FROM chat_logs")
    rows = cursor.fetchall()

    updated_count = 0

    for row_id, query in rows:
        new_cat = classify_intent(query)
        cursor.execute(
            "UPDATE chat_logs SET intent_category = ? WHERE id = ?",
            (new_cat, row_id)
        )
        updated_count += 1

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM chat_logs")
    total_chats = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COALESCE(NULLIF(TRIM(intent_category), ''), 'LAINNYA') AS intent_category,
               COUNT(*)
        FROM chat_logs
        GROUP BY COALESCE(NULLIF(TRIM(intent_category), ''), 'LAINNYA')
        ORDER BY COUNT(*) DESC
    """)
    category_stats = cursor.fetchall()

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_text = f"""
=============================================
   LAPORAN INSIGHT OTOMATIS: TANYA PAK DAHLAN
   Waktu Laporan: {report_time}
=============================================
Total Pengguna : {total_users}
Total Chat     : {total_chats}
Data Diupdate  : {updated_count}

--- Kategori Terpopuler ---
""".strip() + "\n"

    for cat, count in category_stats:
        percent = (count / total_chats * 100) if total_chats > 0 else 0
        report_text += f"{cat:<15}: {count} chat ({percent:.1f}%)\n"

    print(report_text)

    with open("laporan_terakhir.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    conn.close()


if __name__ == "__main__":
    generate_full_report()