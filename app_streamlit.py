import os
import traceback
import streamlit as st
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

st.set_page_config(page_title="Assistente GO", page_icon="🏥", layout="wide")

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "go_books"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-70b-versatile"
TOP_K = 5
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu"
    )
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    import pdfplumber
    all_chunks = []
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        with pdfplumber.open(pdf_file) as pdf:
            book_name = pdf_file.stem
            book_label = "FEBRASGO" if "FEBRASGO" in book_name else "Williams"
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    continue
                words = text.split()
                for i in range(0, len(words), 800):
                    chunk = " ".join(words[i:i + 800])
                    if len(chunk.strip()) > 100:
                        all_chunks.append({
                            "text": chunk.strip(),
                            "metadata": {"source": book_name, "page": page_num, "book": book_label}
                        })

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    for i in range(0, len(all_chunks), 100):
        collection.add(
            ids=ids[i:i+100],
            documents=[c["text"] for c in all_chunks[i:i+100]],
            metadatas=[c["metadata"] for c in all_chunks[i:i+100]]
        )
    return collection


def search_context(collection, query, top_k=TOP_K):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"text": doc, "metadata": meta, "score": 1 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


def build_prompt(query, contexts):
    blocks = []
    for i, ctx in enumerate(contexts, 1):
        m = ctx["metadata"]
        blocks.append(f"[Fonte {i}: {m['book']} - Pagina {m['page']}]\n{ctx['text']}")
    return f"""Voce e um assistente especializado em Ginecologia e Obstetrica.
Use APENAS as informacoes dos trechos abaixo (Tratado FEBRASGO e Williams) para responder.
Cite sempre a fonte (FEBRASGO ou Williams) e a pagina.
Se a informacao nao estiver nos trechos, diga: "Nao encontrei essa informacao nos tratados disponiveis."

=== TRECHOS DOS LIVROS ===
{chr(10).join(blocks)}

=== PERGUNTA ===
{query}

=== RESPOSTA ===
"""


def ask_groq(prompt):
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Voce e um medico especialista em GO. Responda em portugues, cite fontes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    return resp.choices[0].message.content


st.title("Assistente Ginecologia/Obstetrica")
st.caption("Tratado FEBRASGO + Williams Obstetrics - Groq Llama-3.1-70B + RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    collection = get_collection()
    st.sidebar.success(f"Base carregada: {collection.count()} chunks")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar base: {e}")
    st.sidebar.code(traceback.format_exc())
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("### Fontes")
    st.markdown("- Tratado de Ginecologia da FEBRASGO")
    st.markdown("- Williams Obstetrics (Ginecologia de Williams)")
    st.markdown("---")
    st.markdown(f"Modelo: `{GROQ_MODEL}`")
    if st.button("Limpar historico"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte sobre GO (ex: criterios de pre-eclampsia...)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            contexts = search_context(collection, prompt)
            if not contexts:
                answer = "Nao encontrei trechos relevantes nos tratados."
            else:
                answer = ask_groq(build_prompt(prompt, contexts))
            st.markdown(answer)
            if contexts:
                with st.expander("Fontes utilizadas"):
                    for i, ctx in enumerate(contexts, 1):
                        m = ctx["metadata"]
                        st.markdown(f"**{i}. {m['book']}** - Pagina {m['page']} (relevancia: {ctx['score']:.3f})")
                        st.caption(ctx["text"][:300] + "...")
        except Exception as e:
            answer = f"Erro: {e}"
            st.error(answer)
            st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": answer})
