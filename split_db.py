"""
Divide o banco monolitico chroma_db em um banco por livro.

Motivo: o GitHub bloqueia qualquer arquivo individual acima de 100 MB.
A base unica dos 3 livros chega a ~150 MB. Separando por livro, cada
chroma.sqlite3 fica em ~55-75 MB e o app carrega os 3 em paralelo.

Os embeddings sao REAPROVADOS (col.get -> embeddings), entao nao ha
re-embed: apenas reescrita do sqlite. Minutos, nao horas.
"""

import shutil
from pathlib import Path

import chromadb

BASE = Path(__file__).parent
SRC = BASE / "chroma_db"
COLLECTION = "go_books"
BATCH = 256
LIMIT_MB = 99

TARGETS = {
    "Williams": BASE / "chroma_db_williams",
    "SOGIMIG": BASE / "chroma_db_sogimig",
    "FEBRASGO": BASE / "chroma_db_febrasgo",
}


def dir_mb(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1048576


if not SRC.exists():
    raise SystemExit(f"Fonte {SRC} nao existe. Rode index_local.py primeiro.")

for p in TARGETS.values():
    if p.exists():
        print(f"  limpando {p.name}...")
        shutil.rmtree(p)

print("Abrindo base monolitica...")
src_client = chromadb.PersistentClient(path=str(SRC))
src = src_client.get_collection(COLLECTION)
total = src.count()
print(f"  {total} chunks\n")

all_data = src.get(include=["documents", "metadatas", "embeddings"])
books = [m["book"] for m in all_data["metadatas"]]
unknown = set(books) - set(TARGETS)
if unknown:
    raise SystemExit(f"Livro desconhecido no metadata: {unknown}. Ajuste TARGETS.")

print("Criando colecoes...")
clients, cols = {}, {}
for book, path in TARGETS.items():
    clients[book] = chromadb.PersistentClient(path=str(path))
    cols[book] = clients[book].create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

print("Copiando embeddings por livro...")
done = {b: 0 for b in TARGETS}
for i in range(0, total, 2000):
    sl = slice(i, min(i + 2000, total))
    docs = all_data["documents"][sl]
    metas = all_data["metadatas"][sl]
    embs = all_data["embeddings"][sl]
    ids = all_data["ids"][sl]

    groups = {}
    for j, b in enumerate(books[i:i + 2000]):
        groups.setdefault(b, []).append(j)

    for book, idxs in groups.items():
        buf_d, buf_m, buf_e, buf_i = [], [], [], []
        for j in idxs:
            buf_d.append(docs[j])
            buf_m.append(metas[j])
            buf_e.append(embs[j])
            buf_i.append(f"{book}_{ids[j]}")
            if len(buf_d) >= BATCH:
                cols[book].add(ids=buf_i, documents=buf_d,
                                metadatas=buf_m, embeddings=buf_e)
                done[book] += len(buf_d)
                buf_d, buf_m, buf_e, buf_i = [], [], [], []
        if buf_d:
            cols[book].add(ids=buf_i, documents=buf_d,
                            metadatas=buf_m, embeddings=buf_e)
            done[book] += len(buf_d)
    print(f"  {min(i + 2000, total)}/{total}")

del src, src_client

print("\n--- RESULTADO ---")
ok = True
for book, path in TARGETS.items():
    c = chromadb.PersistentClient(path=str(path)).get_collection(COLLECTION)
    mb = dir_mb(path)
    status = "OK" if mb < LIMIT_MB else "ACIMA DO LIMITE"
    if mb >= LIMIT_MB:
        ok = False
    print(f"  {book:<10} {c.count():>5} chunks  {path.name:<22} {mb:7.2f} MB  [{status}]")

print(f"\nTotal: {sum(done.values())}/{total} chunks copiados")
print("RESULTADO:", "TUDO ABAixo DE 99 MB - pode fazer push" if ok
      else "ATENCAO: algum banco passou de 99 MB")
