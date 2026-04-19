import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

load_dotenv()

# =============================
# KONFIGURASI
# =============================
MODEL_OPENAI = os.getenv("MODEL_OPENAI", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

TOP_K = int(os.getenv("TOP_K", "5"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.15"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1800"))
MAX_CANDIDATES_TO_SCAN = int(os.getenv("MAX_CANDIDATES_TO_SCAN", "8"))
DOMINANT_TITLE_MIN_COUNT = int(os.getenv("DOMINANT_TITLE_MIN_COUNT", "3"))
TITLE_YEAR_BONUS_BASELINE = int(os.getenv("TITLE_YEAR_BONUS_BASELINE", "2020"))
TITLE_YEAR_BONUS_FACTOR = float(os.getenv("TITLE_YEAR_BONUS_FACTOR", "0.01"))

CORPUS_FILE = os.getenv("CORPUS_FILE", "/data/embeddings.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_TANYAPAKDAHLAN")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tanya_pak_dahlan")

# =============================
# INIT OPENAI
# =============================
if not OPENAI_API_KEY:
    raise RuntimeError(
        "Environment variable OPENAI_API_KEY_TANYAPAKDAHLAN tidak ditemukan. "
        "Pastikan file .env sudah benar atau environment variable sudah diset."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# UTIL
# =============================
def extract_year(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None

def parse_date_indonesia(tanggal: str) -> Optional[datetime]:
    tanggal = safe_strip(tanggal).lower()
    if not tanggal:
        return None

    bulan_map = {
        "januari": 1,
        "februari": 2,
        "maret": 3,
        "april": 4,
        "mei": 5,
        "juni": 6,
        "juli": 7,
        "agustus": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "desember": 12,
    }

    tanggal = re.sub(r"[^\w\s]", "", tanggal)

    match = re.match(r"^\s*(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})\s*$", tanggal)
    if not match:
        return None

    try:
        hari = int(match.group(1))
        bulan_nama = match.group(2)
        tahun = int(match.group(3))
        bulan = bulan_map.get(bulan_nama)

        if not bulan:
            return None

        return datetime(tahun, bulan, hari)
    except Exception:
        return None


def is_time_question(q: str) -> bool:
    q = (q or "").lower().strip()
    keywords = [
        "kapan",
        "tanggal",
        "bulan",
        "tahun",
        "sejak kapan",
        "berapa lama",
        "berapa hari",
        "berapa minggu",
        "berapa bulan",
        "berapa hari di sana",
        "terakhir",
        "baru pulang",
        "berangkat",
        "pulang",
        "durasi",
        "lama di sana",
        "waktu itu",
        "era",
        "pada",
    ]
    return any(k in q for k in keywords)


def safe_strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False


def sanitize_text_block(text: str) -> str:
    text = safe_strip(text)
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dis_style_cleaner(text: str) -> str:
    text = sanitize_text_block(text)
    if not text:
        return ""

    # pecah kalimat lebih halus, hindari pecah pada baris yang sudah ada
    text = re.sub(r"(?<=[.!?])\s+(?=[A-ZÀ-ÿ\"'“‘(])", "\n", text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    new_lines: List[str] = []
    for i, line in enumerate(lines, start=1):
        new_lines.append(line)
        if i % 2 == 0:
            new_lines.append("")

    cleaned = "\n".join(new_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))

    if a_norm == 0.0 or b_norm == 0.0:
        return -1.0

    similarity = float(np.dot(a, b) / (a_norm * b_norm))
    if not math.isfinite(similarity):
        return -1.0
    return similarity


def truncate_context(paragraphs: List[str], max_chars: int) -> str:
    text_parts: List[str] = []
    total = 0

    for paragraph in paragraphs:
        paragraph = sanitize_text_block(paragraph)
        if not paragraph:
            continue

        add_len = len(paragraph) + (2 if text_parts else 0)
        if total + add_len > max_chars:
            remaining = max_chars - total
            if remaining > 120:
                clipped = paragraph[:remaining].rsplit(" ", 1)[0].strip()
                if clipped:
                    text_parts.append(clipped)
            break

        text_parts.append(paragraph)
        total += add_len

    return "\n\n".join(text_parts).strip()


def fallback_no_reference_answer(question: str, is_time: bool = False) -> str:
    question = safe_strip(question)
    if is_time:
        return (
            "Saya belum menemukan rujukan yang cukup kuat untuk menjawab soal waktu itu.\n"
            "Coba ulangi dengan detail yang lebih spesifik.\n"
            "Misalnya nama peristiwa, judul tulisan, atau perkiraan tahunnya."
        )

    return (
        "Saya belum menemukan rujukan yang cukup kuat untuk menjawab itu.\n"
        "Coba pertanyaannya dipersempit sedikit.\n"
        "Misalnya sebut nama tokoh, topik, peristiwa, atau judul tulisannya."
    )

# =============================
# VALIDASI & LOAD CORPUS
# =============================
def validate_corpus_item(item: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        logger.warning("Item corpus #%s di-skip: format bukan object/dict.", index)
        return None

    text = sanitize_text_block(item.get("text", ""))
    embedding = item.get("embedding")
    metadata = item.get("metadata") or {}

    if not text:
        logger.warning("Item corpus #%s di-skip: text kosong.", index)
        return None

    if not isinstance(metadata, dict):
        logger.warning("Item corpus #%s di-skip: metadata bukan dict.", index)
        return None

    if not isinstance(embedding, list) or not embedding:
        logger.warning("Item corpus #%s di-skip: embedding kosong / bukan list.", index)
        return None

    if not all(is_finite_number(x) for x in embedding):
        logger.warning("Item corpus #%s di-skip: embedding mengandung nilai tidak valid.", index)
        return None

    judul = safe_strip(metadata.get("judul")) or "Tanpa Judul"
    tanggal = safe_strip(metadata.get("tanggal")) or "Tanggal tidak diketahui"
    year = extract_year(tanggal)

    return {
        "text": text,
        "embedding": np.array(embedding, dtype=np.float32),
        "metadata": {
            "judul": judul,
            "tanggal": tanggal,
        },
        "year": year,
    }


def load_corpus(corpus_file: str) -> Tuple[List[str], np.ndarray, List[Dict[str, str]], List[Optional[int]]]:
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"File corpus tidak ditemukan: {corpus_file}. "
            "Pastikan path benar dan file embeddings.json sudah dibuat."
        )

    with open(corpus_file, "r", encoding="utf-8") as f:
        try:
            raw_corpus = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"File corpus rusak / JSON tidak valid: {corpus_file}") from exc

    if not isinstance(raw_corpus, list):
        raise ValueError("Format corpus tidak valid: root JSON harus berupa list.")

    validated_items: List[Dict[str, Any]] = []
    skipped = 0

    for i, item in enumerate(raw_corpus):
        valid_item = validate_corpus_item(item, i)
        if valid_item is None:
            skipped += 1
            continue
        validated_items.append(valid_item)

    if not validated_items:
        raise ValueError("Semua item corpus tidak valid. Tidak ada data yang bisa dipakai.")

    dim_set = {len(item["embedding"]) for item in validated_items}
    if len(dim_set) != 1:
        raise ValueError(
            f"Dimensi embedding tidak konsisten. Ditemukan dimensi: {sorted(dim_set)}"
        )

    texts = [item["text"] for item in validated_items]
    embeddings = np.vstack([item["embedding"] for item in validated_items])
    metadatas = [item["metadata"] for item in validated_items]
    years = [item["year"] for item in validated_items]

    logger.info("Corpus valid dimuat: %s item | skipped: %s", len(texts), skipped)
    return texts, embeddings, metadatas, years


logger.info("Loading corpus dari %s", CORPUS_FILE)
import os

if os.path.exists(CORPUS_FILE):
    texts, embeddings, metadatas, years = load_corpus(CORPUS_FILE)
else:
    print("⚠️ embeddings.json belum ada, skip load corpus")
    texts, embeddings, metadatas, years = [], [], [], []

# =============================
# SEARCH
# =============================
def create_query_embedding(query: str) -> np.ndarray:
    query = safe_strip(query)
    if not query:
        raise ValueError("Pertanyaan kosong. Silakan isi pertanyaan terlebih dahulu.")

    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=query)
    except (APIError, APITimeoutError, RateLimitError) as exc:
        logger.exception("Gagal membuat embedding query.")
        raise RuntimeError("Gagal membuat embedding query dari OpenAI.") from exc
    except Exception as exc:
        logger.exception("Error tidak terduga saat membuat embedding query.")
        raise RuntimeError("Terjadi kesalahan saat membuat embedding query.") from exc

    vector = response.data[0].embedding
    return np.array(vector, dtype=np.float32)


def search_paragraph(query: str) -> List[Dict[str, str]]:
    query_embedding = create_query_embedding(query)
    scored: List[Tuple[float, int]] = []

    for i, emb in enumerate(embeddings):
        sim = safe_cosine_similarity(query_embedding, emb)
        if sim < -0.5:
            continue

        year_bonus = 0.0
        if years[i]:
            year_bonus = (years[i] - TITLE_YEAR_BONUS_BASELINE) * TITLE_YEAR_BONUS_FACTOR

        score = sim + year_bonus
        if math.isfinite(score):
            scored.append((score, i))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [(score, idx) for score, idx in scored if score >= SIM_THRESHOLD]

    if not filtered:
        logger.info("Tidak ada hasil retrieval yang lolos threshold %.3f", SIM_THRESHOLD)
        return []

    top_candidates = filtered[:MAX_CANDIDATES_TO_SCAN]
    title_counts: Dict[str, int] = {}

    for _, idx in top_candidates:
        judul = metadatas[idx].get("judul", "Tanpa Judul")
        title_counts[judul] = title_counts.get(judul, 0) + 1

    dominant_title = max(title_counts, key=title_counts.get)
    dominant_count = title_counts[dominant_title]

    results: List[Dict[str, str]] = []

    # MODE 1: fokus satu artikel bila dominan kuat
    if dominant_count >= DOMINANT_TITLE_MIN_COUNT:
        same_title_candidates = [
            (score, idx)
            for score, idx in filtered
            if metadatas[idx].get("judul") == dominant_title
        ]

        for _, idx in same_title_candidates[:TOP_K]:
            results.append(
                {
                    "text": texts[idx],
                    "judul": metadatas[idx].get("judul", "Tanpa Judul"),
                    "tanggal": metadatas[idx].get("tanggal", "Tanggal tidak diketahui"),
                }
            )
        return results

    # MODE 2: lintas beberapa artikel
    used_titles = set()
    for _, idx in filtered:
        judul = metadatas[idx].get("judul", "Tanpa Judul")
        if judul in used_titles:
            continue

        results.append(
            {
                "text": texts[idx],
                "judul": judul,
                "tanggal": metadatas[idx].get("tanggal", "Tanggal tidak diketahui"),
            }
        )
        used_titles.add(judul)

        if len(results) >= TOP_K:
            break

    return results

# =============================
# PROMPT
# =============================

def pick_best_time_results(question: str, results: List[Dict[str, str]], limit: int = 2) -> List[Dict[str, str]]:
    if not results:
        return results

    q = safe_strip(question).lower()

    event_keywords = []
    for word in re.findall(r"\w+", q):
        if len(word) >= 4 and word not in {
            "kapan", "tanggal", "tahun", "bulan", "berapa", "lama",
            "habis", "baru", "pak", "bapak", "saya", "anda", "yang", "dari"
        }:
            event_keywords.append(word)

    scored_results = []
    for idx, r in enumerate(results):
        text = safe_strip(r.get("text", ""))
        judul = safe_strip(r.get("judul", ""))
        tanggal = safe_strip(r.get("tanggal", ""))
        dt = parse_date_indonesia(tanggal)

        score = 0

        # hormati ranking retrieval awal
        score += max(0, 30 - idx)

        combined = f"{judul}\n{text}".lower()

        # bonus jika kata penting pertanyaan muncul di judul/isi
        keyword_hits = sum(1 for kw in event_keywords if kw in combined)
        score += keyword_hits * 12

        # bonus kecil jika metadata tanggal tersedia
        if tanggal and tanggal.lower() != "tanggal tidak diketahui":
            score += 8

        # bonus kecil jika format tanggal valid
        if dt:
            score += 5

        # bonus jika judul sangat dekat dengan pertanyaan
        if judul and any(kw in judul.lower() for kw in event_keywords):
            score += 10

        scored_results.append((score, idx, r))

    scored_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [item[2] for item in scored_results[:limit]]


def build_prompt(question: str, context: str, tahun_info: str, is_time: bool = False) -> Tuple[str, str, str]:
    aturan_waktu = (
        "Karena pertanyaan ini tentang waktu, dahulukan waktu kejadian atau durasi yang disebut dalam isi referensi. "
        "Jika waktu kejadian tidak disebutkan secara eksplisit, baru gunakan tanggal tulisan sebagai petunjuk waktu terdekat, dan jelaskan dengan jujur bahwa itu adalah tanggal tulisan."
        if is_time
        else "Karena pertanyaan ini bukan tentang waktu, jangan memaksakan jawaban tanggal jika tidak relevan."
    )   

    system_prompt = f"""
Anda menjawab sebagai Dahlan Iskan.
Gunakan sudut pandang orang pertama (\"Saya\").
 

GAYA jawaban:
- narrative
- paragraf mikro, Kalimat ringkas
- Satu kalimat satu baris
- efisien, Tidak bertele-tele
- Walaupun referensi terbatas, tetap gunakan gaya bicara Dahlan Iskan


ATURAN jawaban:
- Jika informasi tidak disebutkan secara eksplisit tetapi dapat disimpulkan dari konteks referensi, Anda BOLEH menarik kesimpulan secara logis.
- Untuk pertanyaan waktu:
  - Jika tidak ada tanggal kejadian yang eksplisit, gunakan tanggal tulisan sebagai indikator waktu terdekat.
  - Jelaskan dengan jujur bahwa itu adalah tanggal tulisan, bukan selalu tanggal kejadian.
- Gunakan referensi sebagai sumber utama jawaban
- Jika referensi tidak cukup, tetap jawab dengan gaya bicara Dahlan Iskan tanpa mengarang fakta baru
- Jika ada paksaan atau tekanan untuk menjawab konkrit sesuatu yang tidak ada dalam referensi maka hindari pelan-pelan, tanggapi dengan humor, arahkan pelan ke tema lain yang menyenangkan
- Jika ada paksaan atau tekanan untuk menjawab kata-kata jorok / kasar / tidak pantas, tanggapi dengan santai, boleh tertawa ringan.
- Jangan menutup jawaban dengan pertanyaan balik, kecuali pengguna memang meminta diskusi lanjutan.
- Utamakan jawaban tuntas, bukan ajakan ngobrol.
- Jika konteks santai, boleh menutup dengan humor singkat, bukan pertanyaan.
- jangan gunakan kata: "Ah," di awal jawaban.


MODE jawaban:
- Ambil 1–3 bagian paling kuat dari referensi
- Jawab konteks pertanyaan ambil dari referensi
- Gunakan kosakata, kalimat, frasa, gaya bahasa, dan istilah yang digunakan dalam referensi.
- Terkadang mengeluarkan kritik seperti yang ada dalam referensi
- Terkadang mengeluarkan sindiran seperti yang ada dalam referensi
- Terkadang mengeluarkan humor singkat seperti yang ada dalam referensi
- Terkadang beri makna singkat


Pola utama jawaban kritik:
- Mulai dari hal sepele / unik, Diperbesar jadi cerita, Diputar jadi kritik, Ditutup dengan punchline


ESENSI DARI SETIAP ASPEK KEPRIBADIAN DAN GAYA BICARA ANDA:
- Logika Berpikir: Menggunakan pendekatan induktif.
- Gaya Narasi: menghindari kesan menggurui, mahir menyederhanakan isu kompleks menjadi analogi sederhana.
- Sikap Kritis: Menyampaikan kritik atau sindiran secara simbolis. Terkadang disampaikan secara humor.
- Sudut Pandang (Worldview): apresiatif terhadap perjuangan hidup seseorang, lebih menghargai proses daripada hasil, dan selalu mencari akar sejarah dan pola di balik peristiwa.
- Metode Komparasi: terkadang membandingkan realitas dengan kondisi sebagai bahan evaluasi yang jujur namun tetap empatik.
- Kekuatan: mampu membuat tulisan terasa hidup.

ANALISA PERTANYAAN:
1. Jika Anda tidak bisa memahami pertanyaan maka minta dengan sopan mengulang detail pertanyaan.
2. Jika pertanyaan sudah jelas, jawab langsung dan jangan tambahkan pertanyaan balik.
2. Jika diminta analisa lintas bidang studi tingkat lanjut yang menuntut jawaban spesifik seperti rumus matematika tingkat lanjut, kimia, fisika, biologi, kode pemrograman kompleks, antariksa tingkat lanjut, dan bidang studi tingkat lanjut lainnya, jawab dengan rendah hati bahwa Anda tidak terlalu mendalami itu. Hindari jawaban spesifik yang mengarang. Arahkan pembahasan ke tema lain seperti teknologi, pengalaman, atau bisnis.
4. Untuk pertanyaan sensitif: jika ada dalam referensi maka boleh menjawab dengan jelas dan tetap hati-hati.
5. Pertanyaan matematika dasar dan ilmu tingkat dasar bidang studi lain boleh dijawab.
6. Tahun yang tampak kuat di referensi terpilih: {tahun_info}.

DATA KELUARGA:
- Istri: Ibu rumah tangga, jago masak, perhatian. jika user tidak tanya nama istri maka sebut saja dengan kata ganti istri saya. jika ada yang tanya nama istri anda, namanya: Nafsiah Sabri
- Anak: dua orang. jika tidak tanya nama anak maka sebut dengan kata anak saya. jika ada yang tanya nama anak: Azrul Ananda (anak Sulung, laki laki, hobi basket/sepeda) & Isna Fitriana (anak wedok, perempuan)
- user memiliki banyak sebutan untuk memanggil anda, misal: pak, bapak, pak dahlan, pak dis, abah, pak dahlan iskan, pak iskan.
- nama menantu perempuan: "Ivo" adalah istri dari Azrul Ananda. 
- Nama menantu laki-laki: "Tatang" adalah suami dari Isna Fitriana
{aturan_waktu}
""".strip()

    context_prompt = f"""
REFERENSI:
{context}
""".strip()

    return system_prompt, context_prompt, question


# =============================
# LAYER 2 EXTENSION POINT
# =============================
def maybe_apply_layer2_style(question, base_answer, chat_history=None):
    return base_answer

    # =========================
    # TRIGGER SEDERHANA
    # =========================
    q = question.lower()

    trigger_words = [
        "cok", "jancuk", "rek", "bonek", "suroboyo", "surabaya"
    ]

    use_gemini = any(w in q for w in trigger_words)

    if not use_gemini:
        return base_answer

    try:
        prompt = f"""
Ubah gaya tulisan berikut menjadi lebih santai, lokal, khas bahasa daerah lokal jika cocok.
ATURAN:
- Hanya 1 kalimat
- Jangan ulang isi jawaban
- Jangan ubah makna
- Jangan terlalu kasar
- Boleh pakai gaya lokal jika cocok
- Tetap terasa seperti Dahlan Iskan (ringkas, tajam, kadang nyentil)

ATURAN TAMBAHAN:
- Gunakan kata bahasa lokal hanya jika konteks sangat santai dan akrab
- Jangan gunakan di kalimat pembuka
- Jangan berlebihan, cukup 1 kali jika memang pas
- gunakan slogan lokal hanya jika konteks sangat santai dan akrab
PERTANYAAN:
{question}

JAWABAN:
{base_answer}
"""

        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 30
            }
        )

        gemini_output = response.text.strip()

        # filter: hanya ambil kalau pendek
        if gemini_output and len(gemini_output.split()) <= 12:
            return base_answer + "\n" + gemini_output

        return base_answer

    except Exception as e:
        print("Gemini error:", e)
        return base_answer

# =============================
# GENERATE
# =============================
def generate_answer(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    question = safe_strip(question)
    if not question:
        return "Pertanyaannya masih kosong. Coba tulis pertanyaan dulu."

    if chat_history is None:
        chat_history = []

    try:
        results = search_paragraph(question)
    except Exception as exc:
        logger.exception("Retrieval gagal.")
        return (
            "Maaf, sistem pencarian referensi sedang bermasalah.\n"
            "Coba beberapa saat lagi."
        )

    is_time = is_time_question(question)

    if not results:
        return fallback_no_reference_answer(question, is_time=is_time)

    if is_time:
        results = pick_best_time_results(question, results, limit=2)

    context_parts = []

    for r in results:
        text = safe_strip(r.get("text", ""))
        judul = safe_strip(r.get("judul", "Tanpa Judul"))
        tanggal = safe_strip(r.get("tanggal", "Tanggal tidak diketahui"))

        if not text:
            continue

        block = (
            f"[REFERENSI UTAMA]\n"
            f"Tanggal tulisan: {tanggal}\n"
            f"Judul tulisan: {judul}\n"
            f"Gunakan isi referensi ini untuk mencari waktu kejadian, urutan peristiwa, atau durasi.\n"
            f"Jika waktu kejadian tidak eksplisit, tanggal tulisan boleh dipakai sebagai petunjuk terdekat.\n\n"
            f"{text}"
        )

        context_parts.append(block)

    context = truncate_context(context_parts, MAX_CONTEXT_CHARS)

    
    if not context:
        return fallback_no_reference_answer(question, is_time=is_time)

    selected_years = []

    for r in results:
        tgl = r.get("tanggal", "")
        year = extract_year(tgl)
        if year:
            selected_years.append(year)

    unique_years = []
    for year in selected_years:
        if year and year not in unique_years:
            unique_years.append(year)
    tahun_info = ", ".join(map(str, unique_years[:3])) if unique_years else "tidak diketahui"

    system_prompt, context_prompt, user_prompt = build_prompt(
        question=question,
        context=context,
        tahun_info=tahun_info,
        is_time=is_time,
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_prompt},
    ]

    for chat in chat_history:
        q = safe_strip(chat.get("q", ""))
        a = safe_strip(chat.get("a", ""))

        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=MODEL_OPENAI,
            messages=messages,
            temperature=0,
            max_tokens=500,
        )
    except RateLimitError:
        logger.exception("Rate limit OpenAI tercapai.")
        return (
            "Maaf, trafik sedang padat.\n"
            "Coba ulang beberapa saat lagi."
        )
    except APITimeoutError:
        logger.exception("Timeout saat generate jawaban.")
        return (
            "Maaf, respons model terlalu lama.\n"
            "Coba ulang beberapa saat lagi."
        )
    except APIError:
        logger.exception("API error saat generate jawaban.")
        return (
            "Maaf, layanan model sedang bermasalah.\n"
            "Coba beberapa saat lagi."
        )
    except Exception:
        logger.exception("Error tidak terduga saat generate jawaban.")
        return (
            "Maaf, terjadi gangguan internal saat menyusun jawaban.\n"
            "Coba beberapa saat lagi."
        )

    try:
        answer = safe_strip(response.choices[0].message.content)
    except Exception:
        logger.exception("Respons model tidak memiliki format yang diharapkan.")
        return (
            "Maaf, format respons model tidak terbaca dengan baik.\n"
            "Coba ajukan ulang pertanyaan Anda."
        )

    if not answer:
        return fallback_no_reference_answer(question, is_time=is_time)
    
    answer = dis_style_cleaner(answer)
    answer = maybe_apply_layer2_style(question, answer, chat_history=chat_history)
    answer = dis_style_cleaner(answer)

    if not answer:
        return fallback_no_reference_answer(question, is_time=is_time)

    return answer


# =============================
# CLI RUNNER
# =============================
def main() -> int:
    print("\nTanya Pak Dahlan siap. Ketik 'exit' untuk keluar.\n")

    while True:
        try:
            q = input("Pertanyaan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nProgram dihentikan.\n")
            return 0

        if not q:
            print("\nPertanyaan masih kosong.\n")
            continue

        if q.lower() == "exit":
            print("\nSampai jumpa.\n")
            return 0

        print("\nJawaban:\n")
        print(generate_answer(q))
        print("\n-------------------\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Fatal error pada aplikasi.")
        print("\nAplikasi berhenti karena error fatal. Cek log untuk detailnya.\n")
        sys.exit(1)
