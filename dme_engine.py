"""
Differential Matrix Engine (DME) — Motor de Raciocínio Clínico Obrigatório
Toda query passa por este pipeline antes de responder.
"""

# Identifica a build em execucao. O Streamlit Cloud mostra esta linha no log
# do container: se ela nao aparecer, o container nao fez redeploy.
BUILD = "dme-v1.1-mod24"

import os
import re
import json
import time
import hashlib
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import OrderedDict
from abc import ABC, abstractmethod
import chromadb
from rank_bm25 import BM25Okapi
import httpx
# ---- LLM PROVIDER SELECTION ------------------------------------------------
# LLM_PROVIDER = "groq" | "ollama" | "llamafile"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

# Ollama config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Llamafile config
LLAMAFILE_BASE_URL = os.getenv("LLAMAFILE_BASE_URL", "http://localhost:8080/v1")
LLAMAFILE_MODEL = os.getenv("LLAMAFILE_MODEL", "llama3.1-8b-instruct")

MAX_DIST = 0.62
TOP_K = 10

# ---- FREE-TIER GUARDS (aplica-se a Groq) ----------------------------------
GROQ_RPM_LIMIT = 30
GROQ_TPM_LIMIT = 6000

# Cache LRU em memoria (TTL 1h) - evita repetir chamadas identicas
_CACHE_TTL = 3600
_CACHE_MAX = 500
_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

# Token bucket para rate limit (Groq)
_bucket_tokens = GROQ_RPM_LIMIT
_bucket_last = time.time()

# Contadores de custo (tokens)
_total_prompt_tokens = 0
_total_completion_tokens = 0
_total_calls = 0

# Logger JSON lines no stdout (gratis no Streamlit logs)
_logger = logging.getLogger("dme")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


def _log(event: str, **kwargs):
    payload = {"event": event, "build": BUILD, "ts": time.time(), **kwargs}
    _logger.info(json.dumps(payload, ensure_ascii=False))


def _cache_key(system, user, temperature, max_tokens):
    h = hashlib.sha256()
    h.update(system.encode())
    h.update(b"|")
    h.update(user.encode())
    h.update(("|{}|{}".format(temperature, max_tokens)).encode())
    return h.hexdigest()[:32]


def _cache_get(key):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < _CACHE_TTL:
            _cache.move_to_end(key)
            return val
        del _cache[key]
    return None


def _cache_set(key, value):
    now = time.time()
    _cache[key] = (now, value)
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def _rate_limit_wait():
    global _bucket_tokens, _bucket_last
    now = time.time()
    _bucket_tokens = min(GROQ_RPM_LIMIT, _bucket_tokens + (now - _bucket_last) * 0.5)
    _bucket_last = now
    if _bucket_tokens < 1:
        wait = (1 - _bucket_tokens) / 0.5
        time.sleep(wait)
        _bucket_tokens = 0
    else:
        _bucket_tokens -= 1


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def _count_tokens(system, user):
    return _estimate_tokens(system), _estimate_tokens(user)


# ---- LLM PROVIDER ABSTRACTION ----------------------------------------------
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system, user, temperature=0.0, max_tokens=6000):
        pass

    @abstractmethod
    def is_available(self):
        pass


class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY nao definida")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def is_available(self):
        return bool(GROQ_API_KEY)

    def chat(self, system, user, temperature=0.0, max_tokens=6000):
        global _total_prompt_tokens, _total_completion_tokens, _total_calls

        key = _cache_key(system, user, temperature, max_tokens)
        cached = _cache_get(key)
        if cached:
            _log("cache_hit", model=self.model, provider="groq")
            return cached

        _rate_limit_wait()

        prompt_toks, _ = _count_tokens(system, user)
        _total_prompt_tokens += prompt_toks
        _total_calls += 1

        last = ""
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens,
                )
                last = (resp.choices[0].message.content or "").strip()
                if hasattr(resp, "usage") and resp.usage:
                    _total_prompt_tokens += getattr(resp.usage, "prompt_tokens", 0)
                    _total_completion_tokens += getattr(resp.usage, "completion_tokens", 0)
            except Exception as e:
                last = "__ERRO__ " + str(e)
            if last and not last.startswith("__ERRO__"):
                _cache_set(key, last)
                _log("groq_ok", attempt=attempt+1, prompt_tokens=prompt_toks,
                     cached=False, cached_size=len(_cache))
                return last
        _log("groq_fail", attempts=3, error=last[:200])
        return last


class OllamaProvider(LLMProvider):
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = OLLAMA_MODEL
        self.client = httpx.Client(timeout=120.0)

    def is_available(self):
        try:
            r = self.client.get(self.base_url + "/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, system, user, temperature=0.0, max_tokens=6000):
        key = _cache_key(system, user, temperature, max_tokens)
        cached = _cache_get(key)
        if cached:
            _log("cache_hit", model=self.model, provider="ollama")
            return cached

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }
        try:
            r = self.client.post(self.base_url + "/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            last = data.get("message", {}).get("content", "").strip()
        except Exception as e:
            last = "__ERRO__ " + str(e)

        if last and not last.startswith("__ERRO__"):
            _cache_set(key, last)
            _log("ollama_ok", model=self.model)
            return last
        _log("ollama_fail", error=last[:200])
        return last


class LlamafileProvider(LLMProvider):
    def __init__(self):
        self.base_url = LLAMAFILE_BASE_URL.rstrip("/")
        self.model = LLAMAFILE_MODEL
        self.client = httpx.Client(timeout=120.0)

    def is_available(self):
        try:
            r = self.client.get(self.base_url + "/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, system, user, temperature=0.0, max_tokens=6000):
        key = _cache_key(system, user, temperature, max_tokens)
        cached = _cache_get(key)
        if cached:
            _log("cache_hit", model=self.model, provider="llamafile")
            return cached

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            r = self.client.post(self.base_url + "/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            last = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = "__ERRO__ " + str(e)

        if last and not last.startswith("__ERRO__"):
            _cache_set(key, last)
            _log("llamafile_ok", model=self.model)
            return last
        _log("llamafile_fail", error=last[:200])
        return last


def _get_llm_provider():
    if LLM_PROVIDER == "ollama":
        return OllamaProvider()
    if LLM_PROVIDER == "llamafile":
        return LlamafileProvider()
    return GroqProvider()


BASE = Path(__file__).parent
COLLECTION_NAME = "go_books"
# uma base por livro: cada chroma.sqlite3 fica abaixo do limite de 100 MB do GitHub
SHARD_DIRS = {
    "Williams": BASE / "chroma_db_williams",
    "SOGIMIG": BASE / "chroma_db_sogimig",
    "FEBRASGO": BASE / "chroma_db_febrasgo",
}
LEGACY_DIR = BASE / "chroma_db"


@dataclass
class Shard:
    book: str
    col: Any
    docs: List[str]
    metas: List[Dict]
    offset: int


@dataclass
class Hypothesis:
    diagnosis: str
    probability: float
    severity: float  # 1-5
    urgency: float   # 1-5
    compatibility: float
    exclusion_strength: float
    layer: str  # A, B, C, D
    system: str
    favoring: List[str]
    against: List[str]
    missing: List[str]
    best_test: str
    test_limitations: str
    confirmatory_finding: str
    excluding_finding: str
    next_step: str


@dataclass
class ClinicalContext:
    complaint: str
    age: Optional[int] = None
    gestational_age_weeks: Optional[int] = None
    gravida: Optional[int] = None
    para: Optional[int] = None
    vital_signs: Dict[str, Any] = None
    physical_exam: str = ""
    risk_factors: List[str] = None
    labs: Dict[str, Any] = None
    imaging: Dict[str, Any] = None
    is_pregnant: bool = False
    # 'confirmada' | 'incerta' | 'nao' | ''
    pregnancy_status: str = ""
    amenorrhea_weeks: Optional[int] = None
    amenorrhea_days: Optional[int] = None
    fetal_viability: str = ""
    multiples: str = ""


MODULOS_24 = [
    "01 Obstetricia geral e pre-natal de baixo risco",
    "02 Emergencias obstetricas: DPP, previa, eclampsia, hemorragia, sepse, rotura",
    "03 Medicina fetal e gestacao de alto risco",
    "04 Infertilidade e medicina reprodutiva",
    "05 Endocrinologia ginecologica: SOP, anovulacao, amenorreia, hiperprolactinemia",
    "06 Menopausa e terapia hormonal",
    "07 Contracepcao e planejamento familiar",
    "08 Uroginecologia e piso pelvico: incontinencia, prolapso",
    "09 Oncologia ginecolica: malignidade, estadiamento, cirurgia oncologica",
    "10 Mastologia: nodulo mamario, carcinoma de mama",
    "11 Ginecologia pediatrica e do adolescente: menarca, anomalia de Muller",
    "12 Colposcopia e lesoes do tracto genital inferior: HPV, HSIL, cervicite",
    "13 Cirurgia ginecolica: laparoscopia, histeroscopia, miomectomia",
    "14 Endometriose",
    "15 Adenomiose",
    "16 Leiomiomas uterinos",
    "17 Disturbios menstruais e dismenorreia",
    "18 Dor pelvica cronica",
    "19 Infeccoes, PID e DST",
    "20 Aborto espontaneo e perda gestacional recorrente",
    "21 Gestacao ectopica e dor em fossa iliaca",
    "22 Hemorragia uterina anomala",
    "23 Dor aguda ginecologica e abdomen agudo ginecologico",
    "24 Climaterio e saude da mulher idosa",
]


def _modulos_24() -> str:
    """Os 24 modulos de especialidade: cobertura, nao hierarquia.

    Sem lista explicita o LLM so detalha os dominios que ja tem em mente.
    Com a lista, ele e obrigado a checar cada um antes de responder.
    """
    return "\n".join(f"  {m}" for m in MODULOS_24)


class DMEEngine:
    def __init__(self, shard_dirs: Optional[Dict[str, Path]] = None):
        # Groq e criado sob demanda: permite testar a recuperacao sem chave
        self._llm = None
        print(f"[dme_engine] build={BUILD} carregado", flush=True)
        self._shard_dirs = shard_dirs

        dirs = shard_dirs if shard_dirs is not None else SHARD_DIRS

        self.shards: List[Shard] = []
        docs: List[str] = []
        metas: List[Dict] = []

        for book, path in dirs.items():
            if not (path / "chroma.sqlite3").exists():
                continue
            client = chromadb.PersistentClient(path=str(path))
            col = client.get_collection(COLLECTION_NAME)
            if col.count() == 0:
                continue
            d = col.get(include=["documents", "metadatas"])
            self.shards.append(Shard(book, col, d["documents"], d["metadatas"],
                                    offset=len(docs)))
            docs.extend(d["documents"])
            metas.extend(d["metadatas"])

        if not self.shards and (LEGACY_DIR / "chroma.sqlite3").exists():
            client = chromadb.PersistentClient(path=str(LEGACY_DIR))
            col = client.get_collection(COLLECTION_NAME)
            d = col.get(include=["documents", "metadatas"])
            self.shards.append(Shard("BASE", col, d["documents"], d["metadatas"], 0))
            docs.extend(d["documents"])
            metas.extend(d["metadatas"])

        if not self.shards:
            raise RuntimeError(
                "Nenhuma base encontrada. Rode 'python index_local.py' e depois "
                "'python split_db.py' no PC e envie as pastas chroma_db_*."
            )

        self.docs = docs
        self.metas = metas
        self.doc_index = {d[:120]: i for i, d in enumerate(docs)}
        try:
            self.bm25 = BM25Okapi([self._norm(d) for d in docs]) if docs else None
        except Exception:
            self.bm25 = None
        if not docs:
            raise RuntimeError(
                "As bases existem mas estao vazias. Rodar no PC: "
                "python index_local.py e depois python split_db.py."
            )

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = _get_llm_provider()
            if not self._llm.is_available():
                raise RuntimeError(
                    "LLM provider '" + LLM_PROVIDER + "' nao disponivel. "
                    "Verifique GROQ_API_KEY / OLLAMA_BASE_URL / LLAMAFILE_BASE_URL."
                )
        return self._llm

    @property
    def books(self) -> List[str]:
        return [s.book for s in self.shards]

    def count(self) -> int:
        return len(self.docs)

    def _shard_of(self, idx: int) -> Shard:
        for s in self.shards:
            if s.offset <= idx < s.offset + len(s.docs):
                return s
        return self.shards[-1]

    def _norm(self, s: str) -> List[str]:
        s = unicodedata.normalize("NFKD", s.lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.findall(r"[a-z0-9]+", s)

    def hybrid_search(self, query: str, k: int = TOP_K,
                        max_dist: float = MAX_DIST) -> List[Dict]:
        """Busca vetorial em cada shard + BM25 global, fundidas por RRF."""
        cands: Dict[tuple, Dict] = {}

        def add(idx, rrf_pts, dist=None):
            key = (self.metas[idx].get("book"), self.metas[idx].get("page"),
                    self.docs[idx][:80])
            if key not in cands:
                cands[key] = {"text": self.docs[idx], "metadata": self.metas[idx],
                                "shard": self._shard_of(idx).book,
                                "dist": 9.9 if dist is None else dist, "rrf": 0.0}
            cands[key]["rrf"] += rrf_pts
            if dist is not None:
                cands[key]["dist"] = min(cands[key]["dist"], dist)

        for s in self.shards:
            n = min(30, len(s.docs))
            if n <= 0:
                continue
            r = s.col.query(query_texts=[query], n_results=n,
                            include=["documents", "metadatas", "distances"])
            for rank, (d, m, dist) in enumerate(
                    zip(r["documents"][0], r["metadatas"][0], r["distances"][0])):
                i = self.doc_index.get(d[:120])
                if i is not None:
                    add(i, 1 / (60 + rank), dist)

        scores = (self.bm25.get_scores(self._norm(query))
                if self.bm25 is not None else [])
        for rank, i in enumerate(list(scores.argsort()[::-1][:30]) if len(scores) else []):
            add(int(i), 1 / (60 + rank))

        out = [c for c in cands.values()
                if c["dist"] <= max_dist or c["rrf"] >= 2 / 90]
        for c in out:
            c["score"] = max(0.0, 1 - c["dist"])
        out.sort(key=lambda c: c["rrf"], reverse=True)
        return out[:k]

    def _chat(self, system: str, user: str, temperature: float = 0.0,
                max_tokens: int = 6000, tries: int = 3) -> str:
        """Chamada LLM via provider (Groq/Ollama/Llamafile) com cache."""
        global _total_prompt_tokens, _total_completion_tokens, _total_calls

        key = _cache_key(system, user, temperature, max_tokens)
        cached = _cache_get(key)
        if cached:
            _log("cache_hit", provider=LLM_PROVIDER)
            return cached

        if LLM_PROVIDER == "groq":
            _rate_limit_wait()

        prompt_toks, _ = _count_tokens(system, user)
        _total_prompt_tokens += prompt_toks
        _total_calls += 1

        last = ""
        for attempt in range(tries):
            try:
                last = self.llm.chat(system, user, temperature, max_tokens)
            except Exception as e:
                last = f"__ERRO__ {e}"
            if last and not last.startswith("__ERRO__"):
                _cache_set(key, last)
                _log("llm_ok", attempt=attempt + 1, provider=LLM_PROVIDER,
                     prompt_tokens=prompt_toks, cached_size=len(_cache))
                return last
        _log("llm_fail", tries=tries, error=last[:200])
        return last

    def get_usage_stats(self) -> Dict[str, Any]:
        """Retorna contadores de uso para monitoring."""
        return {
            "total_calls": _total_calls,
            "prompt_tokens_est": _total_prompt_tokens,
            "completion_tokens_est": _total_completion_tokens,
            "cache_size": len(_cache),
            "provider": LLM_PROVIDER,
        }

    def validate_chunks(self, question: str, ctxs: List[Dict]) -> List[Dict]:
        """LLM julga quais trechos sao realmente relevantes. 1 chamada por lote."""
        if not ctxs:
            return []
        listing = "\n\n".join(
            f"[{i}] {c['metadata'].get('book')} p.{c['metadata'].get('page')}: "
            f"{c['text'][:350]}"
            for i, c in enumerate(ctxs)
        )
        txt = self._chat(
            "Voce e um curador de evidencia medica. Responde de forma curta e "
            "objetiva, apenas com a lista de numeros.",
            "Abaixo estao trechos dos tratados (FEBRASGO, Williams, SOGIMIG). "
            "Quais indices [N] contem informacao REALMENTE relevante para a "
            "pergunta? Responda apenas com os numeros separados por virgula "
            f"ou 'nenhum'.\n\nPERGUNTA: {question}\n\nTRECHOS:\n{listing}",
        ).lower()
        if not txt or txt.startswith("__erro__") or "nenhum" in txt:
            return []
        nums = {int(n) for n in re.findall(r"\d+", txt) if int(n) < len(ctxs)}
        return [c for i, c in enumerate(ctxs) if i in nums]

    def build_dme_prompt(self, context: ClinicalContext, hypotheses: List[Hypothesis],

                        pdf_contexts: Dict[str, List[Dict]],
                        pool: Optional[List[Dict]] = None) -> str:
        hyp_text = "\n".join(
            f"{h.layer}. {h.diagnosis} (prob={h.probability:.0%}, sev={h.severity}, urg={h.urgency}, "
            f"compat={h.compatibility:.0%}, excl={h.exclusion_strength:.0%})"
            for h in hypotheses
        )

        pool = pool if pool is not None else [
            c for ctxs in pdf_contexts.values() for c in ctxs]

        pdf_blocks = ""
        grouped: Dict[str, List[Dict]] = {}
        for c in pool:
            grouped.setdefault(c["metadata"].get("book", "?"), []).append(c)
        for book in sorted(grouped):
            pdf_blocks += f"\n--- {book} ---\n"
            for c in grouped[book][:9]:
                pdf_blocks += (f"\n[{book} p.{c['metadata'].get('page')}] "
                                f"{c['text'][:550]}")
            if len(grouped[book]) > 9:
                pdf_blocks += f"\n[{book} +{len(grouped[book]) - 9} trechos validados]"

        livros = ", ".join(sorted(grouped)) or "nenhum"

        return f"""Voce e o GINECO-OBSTETRICS EXPERT CORE. O DIFFERENTIAL MATRIX ENGINE
v1.0 e o mecanismo que impede o fechamento diagnostico precoce.

CONTEXTO CLINICO
- Queixa: {context.complaint}
- Idade: {context.age}
- GESTACAO: {_pregnancy_block(context)}
- Sinais vitais: {context.vital_signs}
- Exame fisico: {context.physical_exam}
- Fatores de risco: {context.risk_factors}
- Exames laboratoriais: {context.labs}
- Imagem: {context.imaging}

REGRA ESTRUTURAL DE GESTACAO (obrigatoria): a idade gestacional NAO e um
dado complementar, e uma variavel estrutural do differential. A mesma queixa
produz differential completamente diferente conforme a IG. Se a situacao
gestacional for INCERTA ou NAO INFORMADA, isso e o primeiro dado a buscar:
pergunte antes de fechar o raciocinio, e mantenha em aberto a possibilidade
de ectopic e de TPP (teste de gravidez e ultrassonografia).

HIPOTESES GERADAS (5 eixos: probabilidade, gravidade, urgencia,
compatibilidade, exclusao):
{hyp_text}

EVIDENCIA VALIDADA NOS TRATADOS (fontes: {livros}):
{pdf_blocks}

COBERTURA OBRIGATORIA - 24 MODULOS DE ESPECIALIDADE:
{_modulos_24()}

Antes de responder, identifique quais modulos estao em jogo e detalhe
explicitamente os que forem relevantes. Se a queixa se enquadrar em um modulo
que voce nao detailingou, o raciocinio esta incompleto. Em caso de ambiguidade
de enquadramento, considere mais de um modulo.

PIPELINE OBRIGATORIO - execute nesta ordem, sem pular etapas:

1. CARACTERIZACAO CLINICA da queixa e do contexto.
2. REFINAR as hipoteses: recalcule probabilidade, gravidade e urgencia.
3. IDENTIFICAR OS QUE NAO PODEM SER PERDIDOS (camada C), incluindo as
    obstetricas: ectopic, DPP, placenta previa, pre-eclampsia com sinais de
    gravidade, eclampsia, HELLP, hemorragia, sepse materna, rotura uterina,
    sofrimento fetal. E as ginecologicas: torsao anexial, ectopic,
    hemorragia significativa, infeccao pelvica grave, sepse, abdomen agudo.
4. DIFERENCIAIS POR SISTEMA (ginecologico, obstetrico, GI, urinario,
    vascular, endocrinologico, hematologico) com mimetizadores.
5. MATRIZ DE DISCRIMINACAO para cada hipotese relevante (A+B+C).
6. EXAMES DISCRIMINATORIOS com limitacoes: falso-negativo, janela clinica,
    sensibilidade, especificidade, qualidade do exame, probabilidade pre-teste.
    "Exame negativo" NAO e igual a "diagnostico excluido".
7. REAVALIACAO: incorpore todo novo dado. Hipótese sobe, desce, permanece
    ou e descartada. Repita o ciclo.
8. SAFETY CHECK explicito.
9. ANTI-ANCORAGEM: identifique premissas como "e so infeccao", "e apenas
    sangramento menstrual", "a ultrassonografia veio normal", "o beta-hCG
    esta baixo entao nao e ectopic", "ela tem HAS entao a cefaleia e da HAS".
    Para cada uma, pergunte: existe outra hipotese que explique melhor o
    conjunto? Qual diagnostico perigoso ainda nao foi excluido?
10. CONDUTA + FOLLOW-UP.

VOCE NAO E UM CALCULADOR DIAGNOSTICO. Os valores internos servem para
organizar e priorizar. A resposta deve EXPLICAR por que uma hipotese sobe
ou desce. Se os dados nao sustentam fechar o diagnostico, diga:
"Com os dados disponiveis, ainda nao e seguro fechar o diagnostico" e
liste exatamente o que falta.

FORMATO DE SAIDA OBRIGATORIO:

## DIFFERENTIAL MATRIX

**SINDROME:**
[descricao]

**HIPOTESE PRINCIPAL:**
[diagnostico]

**DIFERENCIAIS RELEVANTES:**
1. [diagnostico]
2. [diagnostico]

**NAO PODEM SER PERDIDOS:**
- [diagnostico] - gravidade: [ ] - urgencia: [ ] - por que nao pode ser ignorada
- [diagnostico]

**MIMETIZADORES:**
- [diagnostico]

### MATRIZ DE DISCRIMINACAO
Para cada hipotese relevante (A, B e C), repita:
**HIPOTESE: [nome]**
- A favor: [...]
- Contra: [...]
- Dados que faltam: [...]
- Achado altamente sugestivo: [...]
- Achado incompativel: [...]
- Exame que melhor diferencia: [...]
- Exame que NAO exclui adequadamente: [...]
- Falso-negativo relevante: [...]
- emergencia associada: [...]
- Proximo passo: [...]

### ANTI-ANCORAGEM
| Premissa assumida | Problema | Hipotese alternativa |
|---|---|---|

### SAFETY CHECK
- [ ] Diagnostico potencialmente fatal foi adequadamente excluido?
- [ ] Condicao tempo-dependente abordada?
- [ ] Risco materno avaliado?
- [ ] Risco fetal avaliado?
- [ ] Encaminhamento ou internacao necessarios?
- [ ] Algum diagnostico grave permanece insuficientemente excluido?
**ALERTAS:** [...]

### CONCLUSAO PROVISORIA
**Interpretacao:** [...]
**Nivel de incerteza:** [baixo / moderado / alto]
**Dados necessarios para reduzir a incerteza:** [...]

### CONDUTA + FOLLOW-UP
1. [conduta imediata]
2. [investigacao]
3. [seguimento e reavaliacao]
4. [quando reavaliar e o que dispara escalonamento]

### FONTES CONSULTADAS
[Livro, pagina] - [Livro, pagina] - [Livro, pagina]

REGRAS INEGOCIÁVEIS:
- Baseie-se APENAS na evidencia validada acima. Nao invente.
- Cite livro e pagina em cada afirmacao relevante.
- Se a evidencia for insuficiente, escreva "nos trechos validados nao
encontrei esta informacao" em vez de preencher com suposicao.
- FRAGMENTOS TRUNCADOS: os trechos acima sao cortados em ~550 caracteres.
Se a frase parece incompleta, cortada no meio ou sem a condicao inteira,
isso NAO sustenta inferencia - nao complemente com memoria propria. Declare
que nos trechos validados nao encontrei a informacao.
- FATOR DE RISCO/PROTETOR: o atributo (protetor, neutro, risco) deve seguir
LITERALMENTE o trecho citado, com livro e pagina. Se dois trechos conflitam,
declare o conflito citando as duas fontes. Exemplo ja corrigido: tabagismo
NAO e fator protetor para aborto espontaneo.
- PROIBIDO percentuais para intervencoes ou condutas comparadas (ex: "DIU
reduz X%") e proibido misturar probabilidades de doencas com probabilidades
de intervencoes. Percentual so se estiver EXPLICITAMENTE no trecho, com
livro e pagina.
- CONDUTA nao pode ser apenas "encaminhar / observar / reavaliar". Inclua ao
menos uma opcao CONCRETA: o que fazer agora, com que prazo e qual criterio
de escalonamento.
- Destaque com ⚠️ tudo que nao pode ser perdido.
- Nao prescreva sem qualificar: e suporte ao raciocinio clinico do
profissional, que confirma a conduta.
"""

    def generate_initial_hypotheses(self, context: ClinicalContext) -> List[Hypothesis]:
        """Gera hipóteses iniciais baseadas na queixa + contexto."""
        sys_prompt = """Voce e especialista em Ginecologia e Obstetricia.
Gere hipoteses diagnosticas AMPLAS para a queixa klinica dada.
Organize por SISTEMA (Ginecologico, Obstetrico, GI, Urinario, Vascular, etc.).
Para cada hipotese, estime: probabilidade (0-1), severidade (1-5), urgencia (1-5).
Inclua diagnosticos que NAO PODEM SER PERDIDOS (baixa probabilidade + alta gravidade).
Responda SOMENTE com JSON valido, sem texto em volta, sem markdown.

Formato OBRATORIO:
{"hypotheses": [{"diagnosis": "nome clinico", "probability": 0.5,
"severity": 3, "urgency": 3, "system": "sistema",
"layer": "A", "favoring": ["..."], "against": ["..."], "missing": ["..."],
"best_test": "exame", "test_limitations": "limitacao",
"confirmatory_finding": "achado", "excluding_finding": "achado",
"next_step": "passo"}]}

Camadas: A = mais compativel, B = diferencial relevante,
C = nao pode ser perdida, D = mimetizador."""

        user_prompt = f"""QUEIXA: {context.complaint}
CONTEXTO: idade={context.age}, IG={context.gestational_age_weeks}, G{context.gravida}P{context.para}
Gestante: {context.is_pregnant}
Sinais vitais: {context.vital_signs}
Exame fisico: {context.physical_exam}
Fatores de risco: {context.risk_factors}
Labs: {context.labs}
Imagem: {context.imaging}

JSON:"""

        raw = self._chat(
            "Voce e um gerador de hipoteses clinicas. Responda APENAS com "
            "JSON valido, sem cercas de markdown e sem commentary.",
            user_prompt, temperature=0.3, max_tokens=6000, tries=3,
        )
        return self._parse_hypotheses(raw)

    def _fallback_hypotheses(self, context: ClinicalContext) -> List[Hypothesis]:
        """Estrutura minima caso o LLM falhe: ainda busca evidencia e responde."""
        generic = [("Queixa a esclarecer", "Geral", "B", 3, 3)]
        if context.is_pregnant:
            generic = [
                ("Gestacao a confirmar", "Obstetrico", "A", 4, 4),
                ("Abortamento", "Obstetrico", "C", 4, 5),
                ("Gravidez ectopica", "Obstetrico", "C", 5, 5),
            ]
        return [Hypothesis(
            diagnosis=d, probability=0.2, severity=s, urgency=u,
            compatibility=0.3, exclusion_strength=0.3, layer=l, system=sy,
            favoring=[], against=[], missing=["dados clinicos ausentes"],
            best_test="", test_limitations="", confirmatory_finding="",
            excluding_finding="", next_step="",
        ) for d, sy, l, s, u in generic]

    @staticmethod
    def _parse_hypotheses(raw: str) -> List["Hypothesis"]:
        """Tolera dict, lista pura, cercas de markdown e campos faltando."""
        if not raw:
            return []
        txt = raw.strip()
        txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()

        data = None
        for candidate in (txt, _first_json_object(txt)):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                break
            except Exception:
                continue

        if data is None:
            m = re.search(r"\{.*\}", txt, re.S) or re.search(r"\[.*\]", txt, re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    return []

        if isinstance(data, dict):
            items = (data.get("hypotheses") or data.get("diagnoses")
                    or data.get("differential") or data.get("hipoteses") or [])
        elif isinstance(data, list):
            items = data
        else:
            items = []

        out: List[Hypothesis] = []
        for h in items:
            if isinstance(h, str):
                h = {"diagnosis": h}
            if not isinstance(h, dict):
                continue
            name = h.get("diagnosis") or h.get("nome") or h.get("diagnostico")
            if not name:
                continue

            def num(key, default, lo, hi):
                v = h.get(key, default)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = float(default)
                if hi > 1.5:
                    v = max(lo, min(hi, v))
                return max(lo, min(hi, v))

            def lst(key):
                v = h.get(key)
                if isinstance(v, str):
                    return [v] if v.strip() else []
                return list(v) if isinstance(v, (list, tuple)) else []

            def txt(key):
                v = h.get(key)
                return v if isinstance(v, str) else ""

            out.append(Hypothesis(
                diagnosis=str(name)[:160],
                probability=num("probability", 0.3, 0.0, 1.0),
                severity=num("severity", 3, 1, 5),
                urgency=num("urgency", 3, 1, 5),
                compatibility=num("compatibility", 0.5, 0.0, 1.0),
                exclusion_strength=num("exclusion_strength", 0.5, 0.0, 1.0),
                layer=str(h.get("layer", "B")).strip().upper()[:1] or "B",
                system=txt("system") or "Geral",
                favoring=lst("favoring"), against=lst("against"),
                missing=lst("missing"),
                best_test=txt("best_test"), test_limitations=txt("test_limitations"),
                confirmatory_finding=txt("confirmatory_finding"),
                excluding_finding=txt("excluding_finding"),
                next_step=txt("next_step"),
            ))
        return out


    def search_pdf_for_hypothesis(self, hypothesis: Hypothesis, context: ClinicalContext) -> List[Dict]:
        """Busca trechos relevantes para uma hipótese específica."""
        query = f"{context.complaint} {hypothesis.diagnosis} {hypothesis.system}"
        ctxs = self.hybrid_search(query, k=8, max_dist=0.68)
        validated = self.validate_chunks(f"{context.complaint} {hypothesis.diagnosis}", ctxs)
        return validated

    def _checar_contexto_minimo(self, context: ClinicalContext) -> List[str]:
        """Retorna lista de dados críticos ausentes. Se não vazia, DME deve pausar."""
        faltando = []
        if context.age is None:
            faltando.append("Idade da paciente")
        if context.pregnancy_status == "incerta" and not context.gestational_age_weeks:
            faltando.append("Data da última menstruação ou resultado de β-hCG")
        if context.is_pregnant and context.gestational_age_weeks is None:
            faltando.append("Idade gestacional (semanas)")
        if not context.complaint or len(context.complaint.split()) < 3:
            faltando.append("Descrição do motivo da consulta (sintomas, duração, evolução)")
        return faltando

    def _retrieve_by_module(self, modulo: str, context: ClinicalContext, k: int = 6) -> List[Dict]:
        """Busca forçada no módulo específico para garantir cobertura dos 3 livros."""
        query = f"{context.complaint} {modulo}"
        return self.hybrid_search(query, k=k, max_dist=0.68)

    def run_dme(self, context: ClinicalContext) -> str:
        """Pipeline completo DME. Retorna JSON se precisar de esclarecimento."""
        # 0. Checa contexto mínimo antes de qualquer processamento
        faltando = self._checar_contexto_minimo(context)
        if faltando:
            import json
            return json.dumps({
                "tipo": "ESCLARECIMENTO",
                "mensagem": "Para orientar com segurança, preciso das seguintes informações:",
                "itens": faltando,
                "contexto_atual": {
                    "idade": context.age,
                    "gestacao": context.pregnancy_status,
                    "IG": context.gestational_age_weeks,
                    "queixa": context.complaint,
                }
            }, ensure_ascii=False)

        # 1. Hipóteses iniciais
        hypotheses = self.generate_initial_hypotheses(context)
        if not hypotheses:
            Say = "motor indisponivel"
            hypotheses = self._fallback_hypotheses(context)

        # 2. Busca nos PDFs: 1 query por SISTEMA (agrupa hipoteses) + 1 query
        #    pela queixa pura. Garante cobertura dos 3 livros sem explodir
        #    o numero de chamadas ao LLM.
        by_system: Dict[str, List[str]] = {}
        for h in hypotheses:
            by_system.setdefault(h.system or "Geral", []).append(h.diagnosis)

        queries: List[str] = [context.complaint]
        for system, diags in by_system.items():
            queries.append(f"{context.complaint} {system} {' '.join(diags[:4])}")

        pool: List[Dict] = []
        seen = set()
        for q in queries[:6]:
            for c in self.hybrid_search(q, k=8, max_dist=0.68):
                key = (c["metadata"].get("book"), c["metadata"].get("page"),
                        c["text"][:80])
                if key not in seen:
                    seen.add(key)
                    pool.append(c)

        # 3. Uma unica validacao para todo o pool
        validated = self.validate_chunks(context.complaint, pool[:40])
        by_book: Dict[str, List[Dict]] = {}
        for c in validated:
            by_book.setdefault(c["metadata"].get("book", "?"), []).append(c)
        pdf_contexts = by_book if by_book else {}

        # 4. Prompt DME completo
        prompt = self.build_dme_prompt(context, hypotheses, pdf_contexts,
                                        pool=validated)

        # 5. Resposta final DME
        return self._chat(
            "Voce e o DME - Differential Matrix Engine. Siga o pipeline "
            "rigorosamente. Responda em portugues, no formato Markdown "
            "obrigatorio, citando livro e pagina. Nunca invente informacao "
            "ausente dos trechos.",
            prompt, temperature=0.2, max_tokens=12000,
        )


def _first_json_object(txt: str) -> str:
    """Extrai o primeiro objeto/array JSON balanceado de um texto."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(txt):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    return txt[start:i + 1]
    return ""


_AMENORRHEA = re.compile(
    r"(falta\s+de\s+menstrua|amenorre|nao\s+menstrua|sem\s+menstrua"
    r"|atraso\s+menstru|menstrua\w*\s+(?:ha|à)\s+\d)"
)
_GESTANTE = re.compile(
    r"\b(gestante|gesta[çc][ãa]o|gr[aá]vida|grav[aá]vida|"
    r"\bde\s+\d+\s+semanas\s+de\s+gesta|com\s+\d+\s+semanas)\b"
)


def parse_clinical_input(text: str) -> ClinicalContext:
    """Extrai idade, IG, G/P e situacao gestacional.

    Cuidado clinico: 'falta de menstruacao ha 4 semanas' NAO e 'gestante de
    4 semanas'. Sao situacoes opostas - a primeira suggests amenorreia, a
    segunda uma gestacao，年轻. Por isso nao assumimos gestacao por causa da
    palavra 'semanas' isolada.
    """
    ctx = ClinicalContext(complaint=text)
    low = _strip_accents(text.lower())

    m_age = re.search(r"(\d{1,2})\s*anos?", low)
    if m_age:
        ctx.age = int(m_age.group(1))

    m_gp = re.search(r"\bg\s*(\d)\s*p\s*(\d)\b", low)
    if m_gp:
        ctx.gravida, ctx.para = int(m_gp.group(1)), int(m_gp.group(2))

    tem_amenorreia = bool(_AMENORRHEA.search(low))
    cita_gestante = bool(_GESTANTE.search(low)) or bool(m_gp)

    # --- 1. IG declarada explicitamente ---
    m_ig = None
    for padrao in (
        r"\big\s*(?:de)?\s*:?\s*(\d{1,2})\s*(?:sem|s\b)",
        r"(\d{1,2})\s*sem(?:anas)?\s+de\s+gesta",
        r"gesta\w*\s+(?:de|com)\s+(\d{1,2})\s*sem",
        r"gestante\s+(\d{1,2})\s*sem",
        r"(\d{1,2})\s*sem(?:anas)?\s+(?:de\s+gestacao|gestacional)",
    ):
        m_ig = re.search(padrao, low)
        if m_ig:
            break

    # --- 2. 'ha/com N semanas' sem rotulo: depende do contexto ---
    # 'G2P1 ha 30 semanas'   -> IG 30   (G/P prova que esta gestante)
    # 'falta de menstruacao ha 4 semanas' -> amenorreia de 4 semanas
    if not m_ig:
        m_num = re.search(r"(?:ha|à|com|desde|faz)\s+(\d{1,2})\s*(sem|dias|m[eê]s)", low)
        if m_num:
            n, un = int(m_num.group(1)), m_num.group(2)
            if cita_gestante and not tem_amenorreia:
                m_ig = n
            elif un.startswith("sem"):
                ctx.amenorrhea_weeks = n
            elif un.startswith("dia"):
                ctx.amenorrhea_days = n
            else:
                ctx.amenorrhea_days = n * 30

    # --- 2b. 'N semanas' isolado quando ha G/P (ex: 'G2P1 32 semanas') ---
    if not m_ig and m_gp:
        m_bare = re.search(r"(\d{1,2})\s*sem(?:anas)?\b", low)
        if m_bare:
            m_ig = int(m_bare.group(1))

    if isinstance(m_ig, int):
        ctx.gestational_age_weeks = m_ig
    elif m_ig is not None and hasattr(m_ig, "group"):
        ctx.gestational_age_weeks = int(m_ig.group(1))
    if ctx.gestational_age_weeks:
        ctx.is_pregnant = True

    # --- 3. 'atraso/amenorreia de N semanas' ---
    if not ctx.amenorrhea_weeks and not ctx.amenorrhea_days:
        m_am = re.search(
            r"(?:atraso\s+menstru\w*|amenorre\w*|falta\s+de\s+menstru\w*)"
            r"[^.\d]{0,12}?(\d{1,3})\s*(sem|dias|m[eê]s)", low)
        if m_am:
            n, un = int(m_am.group(1)), m_am.group(2)
            if un.startswith("sem"):
                ctx.amenorrhea_weeks = n
            elif un.startswith("dia"):
                ctx.amenorrhea_days = n
            else:
                ctx.amenorrhea_days = n * 30

    if cita_gestante and not tem_amenorreia:
        ctx.is_pregnant = True

    # Regra clinica: se ha amenorreia e a IG nao foi declarada, a gestacao
    # e INCERTA. Marcamos para o DME perguntar em vez de assumir.
    if tem_amenorreia and not ctx.gestational_age_weeks:
        ctx.pregnancy_status = "incerta"

    return ctx


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _pregnancy_block(ctx: "ClinicalContext") -> str:
    """Bloco de gestacao para o prompt. Explicita o que falta."""
    parts = []
    if ctx.gravida is not None or ctx.para is not None:
        parts.append(f"G{ctx.gravida or '?'}P{ctx.para or '?'}")

    status = ctx.pregnancy_status
    if ctx.gestational_age_weeks:
        parts.append(f"IG {ctx.gestational_age_weeks} semanas (confirmada)")
    elif status == "incerta":
        parts.append("GESTACAO INCERTA - nao confirmada. Nao assumir gestante.")
    elif ctx.is_pregnant:
        parts.append("gestante, IG NAO INFORMADA - dado essencial ausente")
    else:
        parts.append("nao ha indicacao de gestacao no relato")

    if ctx.amenorrhea_weeks:
        parts.append(f"amenorreia ha {ctx.amenorrhea_weeks} semanas")
    elif ctx.amenorrhea_days:
        parts.append(f"amenorreia ha {ctx.amenorrhea_days} dias")
    if ctx.multiples:
        parts.append(f"gestacao {ctx.multiples}")
    if ctx.fetal_viability:
        parts.append(f"vitalidade fetal: {ctx.fetal_viability}")
    return " | ".join(parts)


# CLI para teste
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Caso clínico: ")
    ctx = parse_clinical_input(query)
    engine = DMEEngine()
    print("Executando DME...")
    result = engine.run_dme(ctx)
    print("\n" + "="*60)
    print(result)
