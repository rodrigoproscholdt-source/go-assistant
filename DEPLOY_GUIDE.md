# GO_ASSIST — Guia de Operacao

Assistente clinico de Ginecologia e Obstetricia com **DME** (Differential
Matrix Engine) obrigatorio em toda consulta.

- App: https://go-assistant-rodrigo.streamlit.app/
- Repo: https://github.com/rodrigoproscholdt-source/go-assistant
- Modelo: `openai/gpt-oss-120b` (Groq)
- Embeddings: ONNX local do ChromaDB (sem `torch`, sem `sentence-transformers`)

## Fontes indexadas

| Livro | Chunks | Base | Tamanho |
|---|---|---|---|
| Ginecologia de Williams, 2 ed (Hoffman) — 1419 p | 2917 | `chroma_db_williams` | 59,9 MB |
| Manual SOGIMIG de Ginecologia e Obstetricia, 6 ed — 1079 p | 2102 | `chroma_db_sogimig` | 46,1 MB |
| Tratado de Ginecologia da FEBRASGO — 2717 p | 2547 | `chroma_db_febrasgo` | 37,6 MB |
| **Total** | **7566** | 3 bases | 143,5 MB |

> **Por que 3 bases e nao uma?** O GitHub bloqueia qualquer arquivo individual
> acima de 100 MB. A base unica ficou com `chroma.sqlite3` de 122 MB e seria
> rejeitada. Dividir por tratado resolve sem perder nenhum chunk.

## Comandos (CMD)

```
status.bat          status completo do sistema
limpar.bat          remove rascunhos e arquivos nao rastreados
finalizar.bat       pipeline automatizado: indexar -> split -> testar -> push
type index_log.txt  log da indexacao
type finalizar_log.txt   log da automacao
```

## Fluxo de trabalho

### 1. Indexar (local, ~40 min)

Rode somente quando os PDFs mudarem ou os parametros de chunk mudarem.

```
python -X utf8 index_local.py
```

Parametros em `index_local.py`:

- `CHUNK_WORDS = 520` — tamanho do chunk
- `OVERLAP_WORDS = 65` — sobreposicao entre chunks
- extracao por colunas (evita linhas embaralhadas) + limpeza de hifenizacao
  e marcadores de pagina

### 2. Dividir por livro (local, ~30 s)

```
python -X utf8 split_db.py
```

Le os embeddings da base monolitica e reescreve 3 bases **sem re-embed**.

### 3. Testar (local, sem LLM)

```
python -X utf8 smoke_test.py
```

Valida que os 3 livros carregam, que a busca alcanca os 3 e que cada base
fica abaixo de 99 MB. Sai com codigo 1 em caso de falha.

### 4. Automatizar tudo (indexar -> split -> teste -> push)

```
finalizar.bat
```

Executa 6 gates e **so faz push se todos passarem**:

1. indexacao concluiu e processou 3 livros
2. `split_db.py` executou sem erro
3. todas as bases abaixo de 99 MB
4. smoke test passou
5. nenhum arquivo >= 99 MB no staged do git
6. push para `origin main`

Log em `finalizar_log.txt`.

## Arquitetura

```
app_streamlit.py      interface Streamlit
dme_engine.py         DME: hipotese -> busca nos 3 livros -> validacao -> matriz
index_local.py        extracao dos PDFs + indexacao (local)
split_db.py           divide a base por livro (local)
smoke_test.py         teste de recuperacao sem LLM
status.bat/ps1        status do sistema
limpar.bat/ps1        limpeza da pasta
finalizar.bat/ps1     pipeline automatizado com gates
```

### Pipeline de uma consulta

1. `parse_clinical_input` extrai idade, IG, G/P, gestacao
2. `generate_initial_hypotheses` — LLM gera hipoteses por sistema (A/B/C/D)
3. busca hibrida: 1 query pela queixa + 1 por sistema
   - vetorial em cada base (ONNX MiniLM, cosseno, `MAX_DIST = 0.62`)
   - BM25 global (`rank_bm25`)
   - fusao por RRF
4. `validate_chunks` — **1 chamada LLM** filtra os trechos irrelevantes
5. `_chat` monta a matriz discriminatoria, safety check e conduta
6. resposta em Markdown com citacao `livro` + `pagina`

Total: 3 chamadas LLM por pergunta (hipoteses, validacao, resposta).

## Notas importantes

- **A chave do Groq nunca vai no codigo.** No Streamlit Cloud:
  `Settings > Secrets > GROQ_API_KEY`. Localmente: variavel de ambiente
  `GROQ_API_KEY`.
- Os PDFs **nao estao no git** (`*.pdf` no `.gitignore`): sao insumo de
  indexacao local. O app em runtime usa apenas as bases.
- Se `DMEEngine` levantar `Nenhuma base encontrada`, faltam as pastas
  `chroma_db_williams/`, `chroma_db_sogimig/`, `chroma_db_febrasgo/`.
- O modelo `llama-3.1-70b-versatile` foi descontinuado no Groq. Nao voltar a
  usa-lo.
