import os
import json
import re

INPUT_FOLDER = "corpus_txt"
OUTPUT_FILE = "data_chunks/chunks.jsonl"

os.makedirs("data_chunks", exist_ok=True)


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    parts = text.split("===ISI_ARTIKEL===")
    metadata_text = parts[0]
    article_text = parts[1] if len(parts) > 1 else ""

    metadata = {}
    lines = metadata_text.splitlines()

    current_key = None
    context_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()

        if ":" in line and not line.startswith(" "):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            metadata[key] = val
            current_key = key
        else:
            if current_key == "KONTEKS":
                cleaned = line.strip()
                if cleaned:
                    context_lines.append(cleaned)

    metadata["KONTEKS_FULL"] = " ".join(context_lines).strip()

    return metadata, article_text.strip()

def clean_penutup_artikel(text, max_tail_chars=28):
    text = safe_text(text)
    if not text:
        return ""

    tail_len = min(max_tail_chars, len(text))
    start_tail = len(text) - tail_len
    tail = text[start_tail:]

    tail_lower = tail.lower()

    kandidat = []

    for marker in ["dis", "dahlan", "("]:
        pos = tail_lower.find(marker)
        if pos != -1:
            kandidat.append(pos)

    if not kandidat:
        return text.strip()

    cut_pos_in_tail = min(kandidat)
    cut_pos_in_text = start_tail + cut_pos_in_tail

    hasil = text[:cut_pos_in_text].rstrip()

    # rapikan sisa tanda baca / spasi di ujung
    hasil = re.sub(r"[ \t]+$", "", hasil)
    return hasil.strip()



def chunk_text(text, max_chars=1100, overlap_paragraphs=2):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = []

    for paragraph in paragraphs:
        current.append(paragraph)
        total_length = sum(len(x) for x in current)

        if total_length > max_chars:
            chunks.append("\n".join(current))
            current = current[-overlap_paragraphs:]

    if current:
        chunks.append("\n".join(current))

    return chunks


def format_tanggal(tanggal_str):
    tanggal_str = safe_text(tanggal_str)
    if not tanggal_str:
        return "Tanggal tidak diketahui"

    bulan = {
        "01": "Januari",
        "02": "Februari",
        "03": "Maret",
        "04": "April",
        "05": "Mei",
        "06": "Juni",
        "07": "Juli",
        "08": "Agustus",
        "09": "September",
        "10": "Oktober",
        "11": "November",
        "12": "Desember"
    }

    try:
        y, m, d = tanggal_str.split("-")
        return f"{int(d)} {bulan[m]} {y}"
    except Exception:
        return tanggal_str


def join_metadata_line(label, value):
    value = safe_text(value)
    if not value:
        return None
    return f"{label}: {value}"


def build_embedding_text(meta, chunk):
    tanggal_raw = safe_text(meta.get("TANGGAL"))
    tanggal = format_tanggal(tanggal_raw)
    judul = safe_text(meta.get("JUDUL"))
    kategori = safe_text(meta.get("KATEGORI"))
    tipe = safe_text(meta.get("TIPE"))
    topik_utama = safe_text(meta.get("TOPIK_UTAMA"))
    sub_topik = safe_text(meta.get("SUB_TOPIK"))
    ringkasan = safe_text(meta.get("RINGKASAN"))
    tag_utama = safe_text(meta.get("TAG_UTAMA"))
    tag_tambahan = safe_text(meta.get("TAG_TAMBAHAN"))
    entitas = safe_text(meta.get("ENTITAS"))
    konteks = safe_text(meta.get("KONTEKS_FULL"))

    lines = []

    lines.append("METADATA UTAMA:")
    lines.append(f"Tulisan ini dibuat oleh Dahlan Iskan pada {tanggal}.")
    if judul:
        lines.append(f'Judul tulisan ini adalah "{judul}".')

    info_tema = []
    info_tema.append(join_metadata_line("Kategori", kategori))
    info_tema.append(join_metadata_line("Tipe", tipe))
    info_tema.append(join_metadata_line("Topik utama", topik_utama))
    info_tema.append(join_metadata_line("Sub topik", sub_topik))

    info_tema = [x for x in info_tema if x]
    if info_tema:
        lines.append("")
        lines.append("KLASIFIKASI TEMA:")
        lines.extend(info_tema)

    info_ringkas = []
    info_ringkas.append(join_metadata_line("Ringkasan", ringkasan))
    info_ringkas.append(join_metadata_line("Tag utama", tag_utama))
    info_ringkas.append(join_metadata_line("Tag tambahan", tag_tambahan))
    info_ringkas.append(join_metadata_line("Entitas", entitas))

    info_ringkas = [x for x in info_ringkas if x]
    if info_ringkas:
        lines.append("")
        lines.append("PETUNJUK TAMBAHAN:")
        lines.extend(info_ringkas)

    if konteks:
        lines.append("")
        lines.append("KONTEKS:")
        lines.append(konteks)

    lines.append("")
    lines.append("ISI CHUNK ARTIKEL:")
    lines.append(chunk.strip())

    return "\n".join(lines).strip()


def main():
    files = sorted(os.listdir(INPUT_FOLDER))
    all_chunks = []

    for file in files:
        if not file.endswith(".txt"):
            continue

        path = os.path.join(INPUT_FOLDER, file)
        meta, article = parse_file(path)
        article = clean_penutup_artikel(article, max_tail_chars=28)

        if not article.strip():
            print(f"⚠️ Lewati file tanpa isi artikel: {file}")
            continue

        chunks = chunk_text(article)

        for i, ch in enumerate(chunks):
            data = {
                "doc_id": meta.get("ID"),
                "judul": meta.get("JUDUL"),
                "tanggal": meta.get("TANGGAL"),
                "chunk_index": i,
                "text": build_embedding_text(meta, ch)
            }
            all_chunks.append(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Selesai! Total chunk: {len(all_chunks)}")
    print(f"📄 Output tersimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()