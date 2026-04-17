import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_FILE = "data_chunks/chunks.jsonl"
OUTPUT_FILE = "data_index/embeddings.json"
EMBED_MODEL = "text-embedding-3-small"

os.makedirs("data_index", exist_ok=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_TANYAPAKDAHLAN")
)


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def main():
    chunks = load_chunks(INPUT_FILE)
    results = []

    total = len(chunks)
    print(f"Total chunk ditemukan: {total}")

    for i, item in enumerate(chunks, start=1):
        text = item["text"]

        print(f"[{i}/{total}] Embedding: {item.get('judul')} | chunk {item.get('chunk_index')}")

        emb = get_embedding(text)

        results.append({
            "text": text,
            "embedding": emb,
            "metadata": {
                "doc_id": item.get("doc_id"),
                "judul": item.get("judul"),
                "tanggal": item.get("tanggal"),
                "chunk_index": item.get("chunk_index")
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"\n✅ Selesai! File embedding tersimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()