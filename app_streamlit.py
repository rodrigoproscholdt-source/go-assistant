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
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-70b-versatile"
TOP_K = 5
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.sidebar.write(f"GROQ: {'OK' if GROQ_API_KEY else 'MISSING'}")
st.sidebar.write(f"PDFs: {len(list(PDF_DIR.glob('*.pdf')))}")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from groq import Groq
    import pdfplumber
except Exception as e:
    st.sidebar.error(f"Import error: {e}")
    st.sidebar.code(traceback.format_exc())
    st.stop()


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, device="cpu"
    )
    try:
        col = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
        if col.count() > 0:
            return col
    except Exception:
        pass

    all_chunks = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    for pdf_file in pdfs:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                book_label = "FEBRASGO" if "FEBRASGO" in pdf_file.name else "Williams"
                n_pages = len(pdf.pages)
                for pn, page in enumerate(pdf.pages, 1):
                    if pn % 50 == 0:
                        st.write(f"{book_label}: pagina {pn}/{n_pages}...")
                    text = page.extract_text()
                    if not text or len(text.strip()) < 50:
                        continue
                    words = text.split()
                    for i in range(0, len(words), 800):
                        chunk = " ".join(words[i:i + 800])
                        if len(chunk.strip()) > 100:
                            all_chunks.append({
                                "text": chunk.strip(),
                                "metadata": {"source": pdf_file.stem, "page": pn, "book": book_label}
                            })
        except Exception as e:
            st.write(f"Erro em {pdf_file.name}: {e}")

    st.write(f"Total: {len(all_chunks)} chunks. Indexando...")

    col = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    for i in range(0, len(all_chunks), 200):
        col.add(
            ids=ids[i:i+200],
            documents=[c["text"] for c in all_chunks[i:i+200]],
            metadatas=[c["metadata"] for c in all_chunks[i:i+200]]
        )
        st.write(f"Indexados: {min(i+200, len(all_chunks))}/{len(all_chunks)}")
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
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
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
    st.sidebar.success(f"Base: {collection.count()} chunks")
except Exception as e:
    st.sidebar.error(f"Erro base: {e}")
    st.sidebar.code(traceback.format_exc())
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
