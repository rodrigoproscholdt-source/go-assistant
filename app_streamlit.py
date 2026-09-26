import os
import re
import traceback
import unicodedata
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Assistente GO", layout="wide")
st.title("Assistente Ginecologia/Obstetrica")
st.caption("Tratado FEBRASGO + Williams - Groq gpt-oss-120b + RAG com validacao")

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "go_books"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 10
MAX_DIST = 0.62
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.sidebar.write(f"GROQ: {'OK' if GROQ_API_KEY else 'MISSING'}")
st.sidebar.write(f"PDFs: {len(list(PDF_DIR.glob('*.pdf')))}")

modo = st.sidebar.radio(
    "Formato da resposta",
    ["Resumo clinico estruturado", "Resposta direta"],
    index=0,
)

if "ready" not in st.session_state:
    st.session_state.ready = False

if not st.session_state.ready:
    st.warning("Clique para carregar a base de dados.")
    if st.button("Carregar base"):
        st.session_state.ready = True
        st.rerun()
    st.stop()

try:
    import chromadb
    from groq import Groq
    from rank_bm25 import BM25Okapi
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()


def _norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


@st.cache_resource
def get_search_index():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(name=COLLECTION_NAME)
    if col.count() == 0:
        raise RuntimeError(
            "Base vazia. Rode 'python index_local.py' no PC e envie o chroma_db."
        )
    data = col.get(include=["documents", "metadatas"])
    bm25 = BM25Okapi([_norm(d) for d in data["documents"]])
    doc_index = {d[:120]: i for i, d in enumerate(data["documents"])}
    return col, data, bm25, doc_index


def hybrid_search(col, data, bm25, doc_index, query, k=TOP_K, max_dist=MAX_DIST):
    r = col.query(query_texts=[query], n_results=30,
                  include=["documents", "metadatas", "distances"])
    cands = {}

    def add(idx, rrf_pts, dist=None):
        key = (data["metadatas"][idx]["book"], data["metadatas"][idx]["page"],
               data["documents"][idx][:80])
        if key not in cands:
            cands[key] = {"text": data["documents"][idx],
                          "metadata": data["metadatas"][idx],
                          "dist": dist if dist is not None else 9.9,
                          "rrf": 0.0}
        cands[key]["rrf"] += rrf_pts
        if dist is not None:
            cands[key]["dist"] = min(cands[key]["dist"], dist)

    for rank, (d, m, dist) in enumerate(zip(r["documents"][0], r["metadatas"][0],
                                             r["distances"][0])):
        i = doc_index.get(d[:120])
        if i is not None:
            add(i, 1 / (60 + rank), dist)

    bs = bm25.get_scores(_norm(query))
    order = bs.argsort()[::-1][:30]
    for rank, i in enumerate(order):
        add(i, 1 / (60 + rank))

    out = [c for c in cands.values() if c["dist"] <= max_dist or c["rrf"] >= 2 / 90]
    for c in out:
        c["score"] = max(0.0, 1 - c["dist"])
    out.sort(key=lambda c: c["rrf"], reverse=True)
    return out[:k]


SECTIONS = [
    ("Definicao", "definicao conceito"),
    ("Fisiopatologia", "fisiopatologia mecanismo"),
    ("Clinica", "sintomas sinais quadro clinico"),
    ("Fatores de risco", "fatores de risco epidemiologia"),
    ("Investigacao / Exames", "exames laboratoriais imagem diagnostico"),
    ("Hipoteses diagnosticas", "suspeita criterios diagnostico"),
    ("Diagnosticos diferenciais", "diferencial diferenciar"),
    ("Conduta / Tratamento", "tratamento conduta manejo"),
]


def search_sections(idx, topic):
    col, data, bm25, doc_index = idx
    all_ctxs, by_section = [], {}
    for name, kw in SECTIONS:
        secs = hybrid_search(col, data, bm25, doc_index, f"{topic} {kw}",
                             k=6, max_dist=0.68)
        by_section[name] = secs
        all_ctxs.extend(secs)
    seen, unique = set(), []
    for c in sorted(all_ctxs, key=lambda c: c["rrf"], reverse=True):
        key = (c["metadata"]["book"], c["metadata"]["page"], c["text"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:16], by_section


def validate_chunks(question, ctxs):
    """LLM julga quais trechos sao realmente relevantes para a pergunta."""
    if not ctxs:
        return []
    listing = "\n\n".join(
        f"[{i}] {c['metadata']['book']} p.{c['metadata']['page']}: {c['text'][:350]}"
        for i, c in enumerate(ctxs)
    )
    g = Groq(api_key=GROQ_API_KEY)
    resp = g.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content":
                   "Abaixo estao trechos de livros de Ginecologia/Obstetricia. "
                   "Quais indices [N] contem informacao REALMENTE relevante para a "
                   "pergunta? Responda apenas com os numeros separados por virgula "
                   "ou 'nenhum'.\n\n"
                   f"PERGUNTA: {question}\n\nTRECHOS:\n{listing}"}],
        temperature=0.0, max_tokens=4000,
    )
    txt = (resp.choices[0].message.content or "").lower()
    if "nenhum" in txt:
        return []
    nums = {int(n) for n in re.findall(r"\d+", txt) if int(n) < len(ctxs)}
    return [c for i, c in enumerate(ctxs) if i in nums]


def build_prompt(query, ctxs):
    blocks = "\n\n".join(
        f"[Fonte {i}: {c['metadata']['book']} - Pagina {c['metadata']['page']}]\n{c['text']}"
        for i, c in enumerate(ctxs, 1)
    )
    return f"""Voce e especialista em Ginecologia e Obstetrica.
Responda APENAS com base nos trechos abaixo (FEBRASGO e Williams), citando livro e pagina.
Se os trechos forem insuficientes, diga: "Nos trechos validados nao encontrei resposta."
Nao invente informacao fora dos trechos.

=== TRECHOS VALIDADOS ===
{blocks}

=== PERGUNTA ===
{query}

=== RESPOSTA ===
"""


def build_structured_prompt(query, ctxs, by_section):
    blocks = "\n\n".join(
        f"[Fonte {i}: {c['metadata']['book']} - Pagina {c['metadata']['page']}]\n{c['text']}"
        for i, c in enumerate(ctxs, 1)
    )
    sec_lines = "\n".join(
        f"- {name}: " + (", ".join(f"{c['metadata']['book']} p.{c['metadata']['page']}"
                                   for c in secs[:4]) or "sem trecho")
        for name, secs in by_section.items()
    )
    return f"""Voce e especialista em Ginecologia e Obstetrica, redigindo um RESUMO CLINICO.
Baseie-se APENAS nos trechos validados abaixo. Cite livro e pagina em cada secao.
Se uma secao nao tiver respaldo nos trechos, escreva "Nos trechos validados nao encontrei esta informacao."

Redija em Markdown com EXATAMENTE estas secoes:

## Definicao
## Fisiopatologia
## Clinica (sinais e sintomas)
## Fatores de risco
## Investigacao (exames)
## Hipoteses diagnosticas
## Diagnosticos diferenciais

Regras:
- Portugues claro, em topicos quando couber.
- Cite (Livro, p. X) ao final de cada informacao relevante.
- Termine com "### Fontes consultadas" listando livro e pagina unicos.
- Nao invente informacao fora dos trechos.

=== SUGESTAO DE FONTES POR SECAO ===
{sec_lines}

=== TRECHOS VALIDADOS ===
{blocks}

=== TOPICO ===
{query}

=== RESUMO ===
"""


def ask_groq(prompt, structured=False):
    g = Groq(api_key=GROQ_API_KEY)
    sys_msg = ("Medico especialista em GO. Responde em portugues citando fontes. "
               "Nunca inventa informacao fora do material fornecido.")
    resp = g.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2, max_tokens=8000,
    )
    return resp.choices[0].message.content


try:
    idx = get_search_index()
    st.success(f"Base: {idx[0].count()} chunks | busca hibrida (vetor+BM25) com validacao LLM")
except Exception as e:
    st.error(f"Erro base: {e}")
    st.code(traceback.format_exc())
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Limpar historico"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte sobre GO..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            col, data, bm25, doc_index = idx
            if modo == "Resumo clinico estruturado":
                ctxs, by_section = search_sections(idx, prompt)
                ctxs = validate_chunks(prompt, ctxs)
                if not ctxs:
                    answer = "Nos trechos validados nao encontrei resposta."
                else:
                    answer = ask_groq(build_structured_prompt(prompt, ctxs, by_section),
                                      structured=True)
            else:
                ctxs = hybrid_search(col, data, bm25, doc_index, prompt)
                ctxs = validate_chunks(prompt, ctxs)
                if not ctxs:
                    answer = "Nos trechos validados nao encontrei resposta."
                else:
                    answer = ask_groq(build_prompt(prompt, ctxs))
            st.markdown(answer)
            with st.expander(f"Fontes validadas ({len(ctxs)})"):
                for i, c in enumerate(ctxs, 1):
                    m = c["metadata"]
                    st.markdown(f"**{i}. {m['book']}** p.{m['page']} (relev={c['score']:.3f})")
                    st.caption(c["text"][:300])
        except Exception as e:
            answer = f"Erro: {e}"
            st.error(answer)
            st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": answer})
