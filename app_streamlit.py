import os
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Assistente GO — DME", layout="wide")
BUILD = "2026-09-26-dme-v1.0"
st.title("Assistente Ginecologia/Obstetrícia — DME Engine")
st.caption(
    f"FEBRASGO + Williams + SOGIMIG | Groq gpt-oss-120b | "
    f"build `{BUILD}`"
)

PDF_DIR = Path(__file__).parent
DB_DIR = Path(__file__).parent / "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

# Exemplos rápidos na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Exemplos de comandos:**")
st.sidebar.code("/caso Mulher 28a, G2P1, 32 sem, dor abdominal 6h, febre 38.2")
st.sidebar.code("/diferencial dor pelvica aguda gestacao 8 sem")
st.sidebar.code("/emergencia hipertensao severa 34 sem proteinuria")
st.sidebar.code("/pre-natal 12 sem gestaçao unica risco habitual")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Descreva o caso clinico (ou use /comando)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Se estamos aguardando esclarecimento, combina com o contexto anterior
            if st.session_state.aguardando_esclarecimento:
                ctx_base = st.session_state.aguardando_esclarecimento
                # Re-parsa combinando a nova info
                texto_combinado = f"{ctx_base['queixa_original']} {prompt}"
                ctx = parse_clinical_input(texto_combinado)
                # Preserva idade se já tinha sido dada
                if ctx_base.get('idade'):
                    ctx.age = ctx_base['idade']
            else:
                ctx = parse_clinical_input(prompt.strip())

            with st.spinner("Executando Differential Matrix Engine..."):
                resposta_raw = dme.run_dme(ctx)

            # Verifica se o DME pediu esclarecimento
            try:
                import json
                resposta_json = json.loads(resposta_raw)
                if isinstance(resposta_json, dict) and resposta_json.get("tipo") == "ESCLARECIMENTO":
                    # Guarda o contexto parcial para a proxima rodada
                    st.session_state.aguardando_esclarecimento = {
                        "queixa_original": prompt,
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
