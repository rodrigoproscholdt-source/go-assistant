"""
Smoke test da recuperacao: valida as 3 bases sem chamar o LLM.
Retorna codigo de saida 0 se tudo estiver correto.
"""

import sys
from collections import Counter

from dme_engine import DMEEngine, COLLECTION_NAME, SHARD_DIRS, LEGACY_DIR
from pathlib import Path

QUERIES = [
    ("obstetrica", "conduta no primeiro trimestre de gestacao"),
    ("obstetrica", "pre-eclampsia com sinais de gravidade"),
    ("ginecologia", "tratamento da endometriose"),
    ("ginecologia", "diagnostico do cancer de mama"),
    ("cirurgia", "cesarean section indications"),
    ("fisiologia", "fisiologia do ciclo menstrual"),
]

falhas = []

print("=" * 62)
print("SMOKE TEST - recuperacao multi-bases")
print("=" * 62)

eng = DMEEngine()
print(f"\nbases carregadas : {len(eng.shards)}")
print(f"chunks totais    : {eng.count()}")
por_livro = Counter(m.get("book") for m in eng.metas)
for b, c in sorted(por_livro.items()):
    print(f"  - {b:<10} {c:>6} chunks")
print(f"fontes distintas : {len(por_livro)}")

esperados = {"Williams", "SOGIMIG", "FEBRASGO"}
if por_livro and not esperados.issubset(set(por_livro)):
    print(f"\n[ERRO] esperava os 3 livros {esperados}, veio {set(por_livro)}")
    falhas.append("livros incompletos")
elif not por_livro:
    falhas.append("nenhum chunk carregado")

print("\n" + "-" * 62)
vistos = set()
for tag, q in QUERIES:
    res = eng.hybrid_search(q, k=6, max_dist=0.72)
    if not res:
        print(f"[VAZIO] {tag:<11} {q}")
        falhas.append(f"sem resultado: {q}")
        continue
    top = res[0]
    vistos.add(top["metadata"].get("book"))
    todos = Counter(c["metadata"].get("book") for c in res)
    print(f"[OK] {tag:<11} {q[:44]:<46} -> "
          f"{top['metadata'].get('book')} p.{top['metadata'].get('page')} "
          f"({top['score']:.3f}) | {dict(todos)}")
    for c in res:
        vistos.add(c["metadata"].get("book"))

print("-" * 62)
print(f"livros alcançados pela busca: {sorted(vistos)}")
faltando = esperados - vistos
if faltando:
    print(f"[ERRO] livros nunca apareceram na busca: {sorted(faltando)}")
    falhas.append("livros sem cobertura na busca")

for book, path in SHARD_DIRS.items():
    mb = (sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
          / 1048576) if Path(path).exists() else -1
    if mb < 0:
        print(f"[ERRO] base ausente: {path.name}")
        falhas.append(f"base ausente {book}")
    elif mb >= 99:
        print(f"[ERRO] {book}: {mb:.1f} MB acima do limite do GitHub")
        falhas.append(f"{book} acima de 99 MB")
    else:
        print(f"[OK] {book:<10} {mb:7.2f} MB")

print("\n" + "=" * 62)
if falhas:
    print("RESULTADO: FALHOU -> " + "; ".join(falhas))
    sys.exit(1)
print("RESULTADO: OK")
sys.exit(0)
