import re
import shutil
from pathlib import Path

import chromadb
import pdfplumber

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "go_books"

CHUNK_WORDS = 520
OVERLAP_WORDS = 65

BOOK_RULES = (
    ("FEBRASGO", "FEBRASGO"),
    ("SOGIMIG", "SOGIMIG"),
    ("WILLIAMS", "Williams"),
)


def book_label(name):
    upper = name.upper()
    for needle, label in BOOK_RULES:
        if needle in upper:
            return label
    return "Williams"


def clean_text(t):
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"Hoffman_\d+\.indd\s*\d+\s*", "", t)
    t = re.sub(r"apostilasmedicina@hotmail\.com", "", t)
    t = re.sub(r"\S*@gmail\.com\S*", "", t)
    t = re.sub(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}", " ", t)
    t = re.sub(r"[\uFFFD\u00C2\u00C3]\S{0,2}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def page_text(page):
    words = page.extract_words()
    if not words:
        return page.extract_text() or ""
    mid = page.width / 2
    straddle = sum(1 for w in words if w["x0"] < mid < w["x1"])
    if straddle / len(words) < 0.02 and len(words) > 80:
        left = page.crop((0, 0, mid, page.height)).extract_text() or ""
        right = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
        return left + "\n" + right
    return page.extract_text() or ""


def chunks_from(text, start_page):
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if len(piece) > 100:
            out.append((piece, start_page))
        i += CHUNK_WORDS - OVERLAP_WORDS
    return out


if DB_DIR.exists():
    print("Limpando chroma_db antigo...")
    shutil.rmtree(DB_DIR)

client = chromadb.PersistentClient(path=str(DB_DIR))
col = client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

total = 0
for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
    book_label_value = book_label(pdf_file.name)
    with pdfplumber.open(pdf_file) as pdf:
        print(f"Abrindo {pdf_file.name} ({len(pdf.pages)} paginas)...")
        b_ids, b_docs, b_metas = [], [], []
        for pn, page in enumerate(pdf.pages, 1):
            raw = page_text(page)
            if not raw or len(raw.strip()) < 50:
                continue
            text = clean_text(raw)
            if len(text) < 80:
                continue
            for piece, pp in chunks_from(text, pn):
                b_ids.append(f"{book_label_value}_{total + len(b_ids)}")
                b_docs.append(piece)
                b_metas.append({"source": pdf_file.stem, "page": pp,
                                "book": book_label_value})
            if len(b_docs) >= 128:
                col.add(ids=b_ids, documents=b_docs, metadatas=b_metas)
                total += len(b_ids)
                b_ids, b_docs, b_metas = [], [], []
                print(f"  Indexados: {total} chunks...")
        if b_docs:
            col.add(ids=b_ids, documents=b_docs, metadatas=b_metas)
            total += len(b_ids)
    print(f"Concluido: {pdf_file.name} ({total} chunks)")

print(f"TOTAL: {col.count()} chunks")

per_book = {}
for m in col.get(include=["metadatas"])["metadatas"]:
    per_book[m["book"]] = per_book.get(m["book"], 0) + 1
print("POR LIVRO: " + " | ".join(f"{k}={v}" for k, v in sorted(per_book.items())))

db_bytes = sum(p.stat().st_size for p in DB_DIR.rglob("*") if p.is_file())
print(f"DISCO: {db_bytes / (1024 * 1024):.2f} MB")

for q in ["rotura prematura de membranas",
          "exames do pre natal de rotina",
          "fisiopatologia da preeclampsia"]:
    r = col.query(query_texts=[q], n_results=3, include=["documents", "metadatas", "distances"])
    print(f"\nTESTE: {q}")
    for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"  [{m['book']} p.{m['page']}] dist={dist:.3f} :: {d[:130]}")

print("\nIndexacao concluida com sucesso!")
