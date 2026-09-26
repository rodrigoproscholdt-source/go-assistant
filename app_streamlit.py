import os
import traceback
import streamlit as st

st.set_page_config(page_title="Assistente GO — DME", layout="wide")
st.title("Assistente Ginecologia/Obstetrícia — DME Engine")
st.caption("FEBRASGO + Williams + SOGIMIG | Groq gpt-oss-120b | Differential Matrix Engine")

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
    st.success(f"✅ DME Engine pronto — Base: {dme.collection.count()} chunks")
except Exception as e:
    st.error(f"Erro DME: {e}")
    st.code(traceback.format_exc())
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Limpar historico"):
    st.session_state.messages = []
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

if prompt := st.chat_input("Descreva o caso clínico (ou use /comando)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Parse comando se houver
            text = prompt.strip()
            ctx = parse_clinical_input(text)

            # Executa DME completo
            with st.spinner("🧠 Executando Differential Matrix Engine..."):
                answer = dme.run_dme(ctx)

            st.markdown(answer)

        except Exception as e:
            st.error(f"Erro: {e}")
            st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": answer})