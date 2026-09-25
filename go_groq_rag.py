import os
import json
import pdfplumber
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

PDF_DIR = Path(r"D:\GO_assit")
DB_DIR = Path(r"D:\GO_assit\chroma_db")
COLLECTION_NAME = "go_books"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extrai texto do PDF com metadados de página."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        book_name = pdf_path.stem
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text or len(text.strip()) < 50:
                continue
            chunks.append({
                "text": text.strip(),
                "metadata": {
                    "source": book_name,
                    "page": page_num,
                    "book": "FEBRASGO" if "FEBRASGO" in book_name else "Williams"
                }
            })
    return chunks


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide texto em chunks com sobreposição."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:
            chunks.append(chunk.strip())
    return chunks


def ingest_pdfs():
    """Processa PDFs e popula ChromaDB."""
    print("[INFO] Extraindo texto dos PDFs...")
    all_chunks = []
    for pdf_file in PDF_DIR.glob("*.pdf"):
        print(f"  Processando {pdf_file.name}...")
        pages = extract_pdf_text(pdf_file)
        for page in pages:
            for chunk_text in split_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": page["metadata"]
                })
    
    print(f"  Total de chunks: {len(all_chunks)}")
    
    print("[INFO] Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu"
    )
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    existing = collection.count()
    if existing > 0:
        print(f"  Base já existe com {existing} documentos. Recriando...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    print("[INFO] Indexando chunks (pode levar alguns minutos na primeira vez)...")
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexando"):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    
    print("[OK] Indexação completa: {collection.count()} chunks")
    return collection


def search_context(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Busca chunks relevantes."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "text": doc,
            "metadata": meta,
            "score": 1 - dist
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


def build_prompt(query: str, contexts: list[dict]) -> str:
    """Constrói prompt com contexto dos livros."""
    context_blocks = []
    for i, ctx in enumerate(contexts, 1):
        meta = ctx["metadata"]
        context_blocks.append(
            f"[Fonte {i}: {meta['book']} - Página {meta['page']}]\n{ctx['text']}"
        )
    
    context_str = "\n\n---\n\n".join(context_blocks)
    
    return f"""Você é um assistente especializado em Ginecologia e Obstetrícia.
Use APENAS as informações dos trechos abaixo (Tratado FEBRASGO e Williams) para responder.
Cite sempre a fonte (FEBRASGO ou Williams) e a página.
Se a informação não estiver nos trechos, diga: "Não encontrei essa informação nos tratados disponíveis."

=== TRECHOS DOS LIVROS ===
{context_str}

=== PERGUNTA ===
{query}

=== RESPOSTA ===
"""


def chat_loop(collection):
    """Loop de chat interativo."""
    client = Groq(api_key=GROQ_API_KEY)
    
    print("\n=== Assistente GO (Groq + RAG Local) ===")
    print("Comandos: /sair /reindex /stats\n")
    
    while True:
        try:
            q = input("\n❓ Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not q:
            continue
        if q == "/sair":
            break
        if q == "/stats":
            print(f"  Chunks indexados: {collection.count()}")
            continue
        if q == "/reindex":
            ingest_pdfs()
            continue
        
        print("[INFO] Buscando...")
        contexts = search_context(collection, q)
        
        if not contexts:
            print("[AVISO] Nenhum trecho relevante encontrado.")
            continue
        
        prompt = build_prompt(q, contexts)
        
        print("[INFO] Gerando resposta...")
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Você é um médico especialista em GO. Responda em português, cite fontes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            answer = resp.choices[0].message.content
            print(f"\n[RESPOSTA]:\n{answer}")
            
            print("\n[FONTES]:")
            for i, ctx in enumerate(contexts, 1):
                m = ctx["metadata"]
                print(f"  {i}. {m['book']} - Página {m['page']} (score: {ctx['score']:.3f})")
        
        except Exception as e:
            print(f"  [ERRO] {e}")


def main():
    if not DB_DIR.exists() or not list(DB_DIR.iterdir()):
        print("[INFO] Primeira execução: indexando PDFs...")
        collection = ingest_pdfs()
    else:
        print("[INFO] Carregando base existente...")
        client = chromadb.PersistentClient(path=str(DB_DIR))
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu"
        )
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
        print(f"  Carregados {collection.count()} chunks")
    
    chat_loop(collection)


if __name__ == "__main__":
    main()
