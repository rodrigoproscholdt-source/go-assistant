import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

API_KEY = os.getenv("GEMINI_API_KEY") or input("Cole sua API Key do Gemini: ").strip()
genai.configure(api_key=API_KEY)

MODEL = "gemini-1.5-flash"
SAFETY = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

def list_files():
    print("\n--- Seus arquivos no Gemini ---")
    for f in genai.list_files():
        print(f"  {f.display_name}  ({f.name})")
    print()

def upload_pdf(path, name):
    print(f"Enviando {name}...")
    f = genai.upload_file(path, display_name=name)
    print(f"  OK: {f.name}")
    return f

def main():
    print("=== Assistente GO (FEBRASGO + Williams) ===")
    print("Comandos: /list /upload /sair\n")

    files = list(genai.list_files())
    febrasgo = next((f for f in files if "FEBRASGO" in f.display_name), None)
    williams = next((f for f in files if "Williams" in f.display_name), None)

    if not febrasgo:
        febrasgo = upload_pdf(r"D:\GO_assit\Tratado de Ginecologia da FEBRASGO.pdf", "FEBRASGO")
    if not williams:
        williams = upload_pdf(r"D:\GO_assit\Ginecologia de Williams - Hoffman et al_ - 2 ed_ 2014 - Pt.pdf", "Williams")

    model = genai.GenerativeModel(MODEL, safety_settings=SAFETY)
    chat = model.start_chat(history=[])

    while True:
        try:
            q = input("\n❓ Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q == "/sair":
            break
        if q == "/list":
            list_files()
            continue
        if q == "/upload":
            path = input("Caminho do PDF: ").strip().strip('"')
            name = input("Nome: ").strip()
            upload_pdf(path, name)
            continue

        prompt = f"""Use APENAS os dois tratados anexados (FEBRASGO e Williams) para responder.
Cite a fonte (FEBRASGO ou Williams) e, se possível, capítulo/seção.
Se a informação não estiver nos livros, diga explicitamente.

Pergunta: {q}"""

        try:
            resp = chat.send_message([prompt, febrasgo, williams])
            print(f"\n📋 Resposta:\n{resp.text}")
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    main()