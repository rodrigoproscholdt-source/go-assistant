import os
import traceback
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Assistente GO — DME", layout="wide")
BUILD = "2026-09-26-dme-v1.1"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
st.title("Assistente Ginecologia/Obstetrícia — DME Engine")
st.caption(
    f"FEBRASGO + Williams + SOGIMIG | LLM: {LLM_PROVIDER} | "
    f"build `{BUILD}`"
)

# ---- HEALTH CHECK (gratis, sem Groq) ----
# Acesse: https://seu-app.streamlit.app/?healthz=1
if st.query_params.get("healthz") == "1":
    try:
        from dme_engine import DMEEngine
        eng = DMEEngine()
        stats = eng.get_usage_stats()
        books = list(eng.books) if hasattr(eng, "books") else []
        st.json({
            "status": "healthy",
            "build": BUILD,
            "chunks": eng.count(),
            "books": books,
            "groq_key": "OK" if os.getenv("GROQ_API_KEY") else "MISSING",
            "usage": stats,
        })
    except Exception as e:
        st.json({"status": "unhealthy", "error": str(e)})
    st.stop()

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.sidebar.write(f"LLM provider: {LLM_PROVIDER}")
st.sidebar.write(f"GROQ: {'OK' if GROQ_API_KEY else 'MISSING'}")
st.sidebar.write(f"PDFs: {len(list(PDF_DIR.glob('*.pdf')))}")

# Modo DME sempre ativo
st.sidebar.info("🧠 **DME Engine ATIVO** — Toda consulta passa pelo pipeline completo")

if "ready" not in st.session_state:
    st.session_state.ready = False

if not st.session_state.ready:
    st.warning("Clique para carregar a base de dados.")
    if st.button("Carregar base"):
        st.session_state.ready = True
        st.rerun()
    st.stop()

try:
    from dme_engine import DMEEngine, parse_clinical_input
except Exception as e:
    st.error(f"Import DME error: {e}")
    st.code(traceback.format_exc())
    st.stop()


@st.cache_resource
def get_dme_engine():
    return DMEEngine()


try:
    dme = get_dme_engine()
    st.success(
        f"DME pronto - {dme.count()} chunks | fontes: {', '.join(dme.books)}"
    )
    st.caption(
        "Busca hibrida (vetor + BM25) com validacao LLM. "
        "Embeddings: ONNX local, sem torch."
    )
    # Stats de uso (free tier monitoring)
    with st.sidebar.expander("📊 Uso Groq (free tier)", expanded=False):
        stats = dme.get_usage_stats()
        st.json(stats)
except Exception as e:
    st.error(f"Erro ao carregar o DME: {e}")
    st.code(traceback.format_exc())
    st.info(
        "Se a mensagem mencionar base vazia ou ausente: as pastas "
        "chroma_db_williams/, chroma_db_sogimig/ e chroma_db_febrasgo/ "
        "precisam estar no repositorio. conferir no Streamlit Cloud > "
        "Settings > Manage app > Deployment > Redeploy."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "aguardando_esclarecimento" not in st.session_state:
    st.session_state.aguardando_esclarecimento = None

# Inicializado aqui para que NUNCA exista NameError, mesmo se o bloco de
# chat for interrompido por excecao ou por uma versao antiga em cache.
answer = "Nao foi possivel completar a analise. Tente novamente."

if st.sidebar.button("Limpar historico"):
    st.session_state.messages = []
    st.session_state.aguardando_esclarecimento = None
    st.rerun()

def _extract_pdf_text(file) -> str:
    try:
        import io
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
            partes = []
            for p in pdf.pages[:20]:
                partes.append(p.extract_text() or "")
        return "\n".join(partes)
    except Exception as e:
        return f"[texto de {file.name} nao extraivel: {e}]"


def montar_entrada(prompt: str) -> str:
    """Combina prompt + HDA + HPP + boxe + exames anexados em um unico texto."""
    blocos = [prompt.strip()]
    hda = (st.session_state.get("hda") or "").strip()
    hpp = (st.session_state.get("hpp") or "").strip()
    boxe = (st.session_state.get("boxe") or "").strip()
    if hda:
        blocos.append("HDA: " + hda)
    if hpp:
        blocos.append("HPP: " + hpp)
    if boxe:
        blocos.append("Exame fisico (boxe): " + boxe)
    for nome, txt in (st.session_state.get("pdf_texts") or {}).items():
        if txt.strip():
            blocos.append("Resultado de exame (%s): %s" % (nome, txt[:6000]))
    uploads = st.session_state.get("uploads") or []
    imgs = [f.name for f in uploads
            if getattr(f, "type", "").startswith("image/")]
    if imgs:
        blocos.append("Imagens de resultados anexadas: " + ", ".join(imgs))
    return "\n\n".join(blocos)


# ---- ENTRADA ESTRUTURADA: HDA / HPP / BOXE / EXAMES ----
st.markdown("#### Dados do caso")
col1, col2, col3 = st.columns(3)
with col1:
    st.text_area(
        "HDA — Historia da Doenca Atual", height=150, key="hda",
        placeholder="Queixa principal, evolucao, tempo, associacoes...",
    )
with col2:
    st.text_area(
        "HPP — Historia Patologica Passada", height=150, key="hpp",
        placeholder="Comorbidades, cirurgias, medicacoes, alergias...",
    )
with col3:
    st.text_area(
        "Boxe — Exame Fisico", height=150, key="boxe",
        placeholder="PA, FC, T, abdome, speculum, toque, pavimento...",
    )

st.markdown("#### Resultados de exames")
uploads = st.file_uploader(
    "Upload de PDF ou imagens (pasta DCIM, laudos, exames)",
    type=["pdf", "png", "jpg", "jpeg", "webp", "gif"],
    accept_multiple_files=True,
    key="uploads",
)

if uploads:
    imgs = [f for f in uploads if getattr(f, "type", "").startswith("image/")]
    pdfs = [f for f in uploads if not getattr(f, "type", "").startswith("image/")]

    # Boxe dedicado a imagens de resultados
    with st.expander("Imagens de resultados (%d)" % len(imgs), expanded=bool(imgs)):
        for f in imgs:
            try:
                st.image(f, caption=f.name)
            except Exception as e:
                st.warning("Nao foi possivel exibir %s: %s" % (f.name, e))

    # Extrai texto dos PDFs (cache por nome)
    if "pdf_texts" not in st.session_state:
        st.session_state.pdf_texts = {}
    ativos = {f.name for f in pdfs}
    for nome in list(st.session_state.pdf_texts):
        if nome not in ativos:
            del st.session_state.pdf_texts[nome]
    for f in pdfs:
        if f.name not in st.session_state.pdf_texts:
            st.session_state.pdf_texts[f.name] = _extract_pdf_text(f)
    if pdfs:
        st.caption(
            "PDFs anexados (%d): texto extraido e injetado no caso."
            % len(pdfs)
        )
        with st.expander("Texto extraido dos PDFs", expanded=False):
            for nome, txt in st.session_state.pdf_texts.items():
                st.markdown("**%s**" % nome)
                st.text(txt[:4000])

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Descreva o caso clinico (ou use /comando)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            entrada = montar_entrada(prompt)

            # Se estamos aguardando esclarecimento, combina com o contexto anterior
            if st.session_state.aguardando_esclarecimento:
                ctx_base = st.session_state.aguardando_esclarecimento
                # Re-parsa combinando a nova info
                texto_combinado = f"{ctx_base['queixa_original']} {entrada}"
                ctx = parse_clinical_input(texto_combinado)
                # Preserva idade se já tinha sido dada
                if ctx_base.get('idade'):
                    ctx.age = ctx_base['idade']
            else:
                ctx = parse_clinical_input(entrada)

            with st.spinner("Executando Differential Matrix Engine..."):
                resposta_raw = dme.run_dme(ctx)

            # Verifica se o DME pediu esclarecimento
            try:
                import json
                resposta_json = json.loads(resposta_raw)
                if isinstance(resposta_json, dict) and resposta_json.get("tipo") == "ESCLARECIMENTO":
                    # Guarda o contexto parcial para a proxima rodada
                    st.session_state.aguardando_esclarecimento = {
                        "queixa_original": entrada,
                        "idade": ctx.age,
                        "faltando": resposta_json["itens"]
                    }
                    # Mostra a pergunta ao usuario
                    msg = resposta_json["mensagem"] + "\n\n" + "\n".join(f"- {i}" for i in resposta_json["itens"])
                    st.markdown(msg)
                    answer = msg
                else:
                    # Resposta normal do DME
                    st.session_state.aguardando_esclarecimento = None
                    st.markdown(resposta_raw)
                    answer = resposta_raw
            except (json.JSONDecodeError, TypeError):
                # Nao e JSON de esclarecimento, trata como resposta normal
                st.session_state.aguardando_esclarecimento = None
                st.markdown(resposta_raw)
                answer = resposta_raw

            with st.expander("Fontes consultadas (resultado da busca)"):
                ctxs = dme.hybrid_search(ctx.complaint, k=10, max_dist=0.68)
                for i, c in enumerate(ctxs, 1):
                    m = c["metadata"]
                    st.markdown(
                        f"**{i}. {m.get('book')}** p.{m.get('page')} "
                        f"(relevancia={c['score']:.3f})"
                    )
                    st.caption(c["text"][:300])

        except Exception as e:
            answer = f"Erro ao processar: {e}"
            st.error(answer)
            st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": answer})
