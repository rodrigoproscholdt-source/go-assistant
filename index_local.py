import shutil
from pathlib import Path

import chromadb
import pdfplumber

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "go_books"

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
            text = page.extract_text()
            if not text or len(text.strip()) < 50:
                continue
            words = text.split()
            for i in range(0, len(words), 800):
                chunk = " ".join(words[i:i + 800])
                if len(chunk.strip()) <= 100:
                    continue
                b_ids.append(f"{book_label}_{pn}_{len(b_ids)}")
                b_docs.append(chunk.strip())
                b_metas.append({"source": pdf_file.stem, "page": pn, "book": book_label})
            if len(b_docs) >= 64:
                col.add(ids=b_ids, documents=b_docs, metadatas=b_metas)
                total += len(b_ids)
                b_ids, b_docs, b_metas = [], [], []
                if total % 500 < 64:
                    print(f"  Indexados: {total} chunks...")
        if b_docs:
            col.add(ids=b_ids, documents=b_docs, metadatas=b_metas)
            total += len(b_ids)
    print(f"Concluido: {pdf_file.name} ({total} chunks)")

print(f"TOTAL: {col.count()} chunks")

r = col.query(query_texts=["rotura prematura de membranas"], n_results=3,
              include=["documents", "metadatas", "distances"])
print("\nTESTE DE BUSCA: rotura prematura de membranas")
for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
    print(f"  [{m['book']} p.{m['page']}] dist={dist:.3f} :: {d[:120]}")

print("\nIndexacao concluida com sucesso!")
