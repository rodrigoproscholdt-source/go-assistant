import os
import traceback
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Assistente GO", layout="wide")
st.title("Assistente Ginecologia/Obstetrica")
st.caption("Tratado FEBRASGO + Williams Obstetrics - Groq Llama-3.1-70B + RAG")

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "go_books"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.sidebar.write(f"GROQ: {'OK' if GROQ_API_KEY else 'MISSING'}")
st.sidebar.write(f"PDFs: {len(list(PDF_DIR.glob('*.pdf')))}")

if "ready" not in st.session_state:
    st.session_state.ready = False

if not st.session_state.ready:
    st.warning("Clique para carregar a base de dados.")
    if st.button("Carregar base"):
        st.session_state.ready = True
        st.rerun()
    st.stop()

# Daqui pra baixo so roda apos o clique
try:
    import chromadb
    from groq import Groq
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(traceback.format_exc())
    st.stop()


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    if col.count() == 0:
        raise RuntimeError(
            "Base vazia. Rode 'python index_local.py' no PC e envie o chroma_db para o GitHub."
        )
    return col


def search(collection, query):
    r = collection.query(query_texts=[query], n_results=TOP_K,
                         include=["documents", "metadatas", "distances"])
    return [{"text": d, "metadata": m, "score": 1 - dist}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


def build_prompt(query, ctxs):
    blocks = "\n\n".join(
        f"[Fonte {i}: {c['metadata']['book']} - Pagina {c['metadata']['page']}]\n{c['text']}"
        for i, c in enumerate(ctxs, 1)
    )
    return f"""Voce e especialista em Ginecologia e Obstetrica.
Use APENAS os trechos abaixo (FEBRASGO e Williams). Cite fonte e pagina.
Se nao encontrar, diga: "Nao encontrei nos tratados disponiveis."

=== TRECHOS ===
{blocks}

=== PERGUNTA ===
{query}

=== RESPOSTA ===
"""


def ask_groq(prompt):
    g = Groq(api_key=GROQ_API_KEY)
    resp = g.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Medico especialista em GO. Responda em portugues, cite fontes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2, max_tokens=2000,
    )
    return resp.choices[0].message.content


try:
    collection = get_collection()
    st.success(f"Base: {collection.count()} chunks")
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
            ctxs = search(collection, prompt)
            if not ctxs:
                answer = "Nao encontrei trechos relevantes."
            else:
                answer = ask_groq(build_prompt(prompt, ctxs))
            st.markdown(answer)
            with st.expander("Fontes"):
                for i, c in enumerate(ctxs, 1):
                    m = c["metadata"]
                    st.markdown(f"**{i}. {m['book']}** p.{m['page']} ({c['score']:.3f})")
                    st.caption(c["text"][:300])
        except Exception as e:
            answer = f"Erro: {e}"
            st.error(answer)
            st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": answer})
