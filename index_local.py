import re
import shutil
from pathlib import Path

import chromadb
import pdfplumber

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "go_books"

CHUNK_WORDS = 360
OVERLAP_WORDS = 30


def clean_text(t):
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"Hoffman_\d+\.indd\s*\d+\s*", "", t)
    t = re.sub(r"apostilasmedicina@hotmail\.com", "", t)
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
    book_label = "FEBRASGO" if "FEBRASGO" in pdf_file.name else "Williams"
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
                b_ids.append(f"{book_label}_{total + len(b_ids)}")
                b_docs.append(piece)
                b_metas.append({"source": pdf_file.stem, "page": pp, "book": book_label})
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

for q in ["rotura prematura de membranas",
          "exames do pre natal de rotina",
          "fisiopatologia da preeclampsia"]:
    r = col.query(query_texts=[q], n_results=3, include=["documents", "metadatas", "distances"])
    print(f"\nTESTE: {q}")
    for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"  [{m['book']} p.{m['page']}] dist={dist:.3f} :: {d[:130]}")

print("\nIndexacao concluida com sucesso!")
