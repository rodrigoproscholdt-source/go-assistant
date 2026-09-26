"""
Differential Matrix Engine (DME) — Motor de Raciocínio Clínico Obrigatório
Toda query passa por este pipeline antes de responder.
"""

import os
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from groq import Groq
import chromadb
from rank_bm25 import BM25Okapi

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_DIST = 0.62
TOP_K = 10

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


class DMEEngine:
    def __init__(self, shard_dirs: Optional[Dict[str, Path]] = None):
        self.groq = Groq(api_key=GROQ_API_KEY)
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
        self.bm25 = BM25Okapi([self._norm(d) for d in docs])

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

        scores = self.bm25.get_scores(self._norm(query))
        for rank, i in enumerate(scores.argsort()[::-1][:30]):
            add(int(i), 1 / (60 + rank))

        out = [c for c in cands.values()
               if c["dist"] <= max_dist or c["rrf"] >= 2 / 90]
        for c in out:
            c["score"] = max(0.0, 1 - c["dist"])
        out.sort(key=lambda c: c["rrf"], reverse=True)
        return out[:k]

    def _chat(self, system: str, user: str, temperature: float = 0.0,
              max_tokens: int = 6000, tries: int = 3) -> str:
        """Chamada Groq com retry: o modelo as vezes devolve content vazio."""
        last = ""
        for attempt in range(tries):
            try:
                resp = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens,
                )
                last = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last = f"__ERRO__ {e}"
            if last and not last.startswith("__ERRO__"):
                return last
        return last

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

        return f"""Você é o DIFFERENTIAL MATRIX ENGINE (DME) — Ginecologia/Obstetrícia.
Siga RIGOROSAMENTE o pipeline DME. Não pule etapas.

CONTEXTO CLÍNICO:
- Queixa: {context.complaint}
- Idade: {context.age}
- Idade gestacional: {context.gestational_age_weeks or 'N/A'} semanas
- Gestação: G{context.gravida}P{context.para}
- Sinais vitais: {context.vital_signs}
- Exame físico: {context.physical_exam}
- Fatores de risco: {context.risk_factors}
- Exames laboratoriais: {context.labs}
- Imagem: {context.imaging}
- Gestante: {'SIM' if context.is_pregnant else 'NÃO'}

HIPÓTESES GERADAS:
{hyp_text}

TRECHOS VALIDADOS DOS TRATADOS (fontes consultadas: {livros}):
{pdf_blocks}

=== PIPELINE OBRIGATÓRIO ===

1. REFINAR HIPÓTESES: Ajuste probabilidade/severidade/urgência baseado no contexto.
2. IDENTIFICAR CANNOT-MISS: Liste diagnósticos que NÃO PODEM SER PERDIDOS (Camada C).
3. MATRIZ DE DISCRIMINAÇÃO: Para cada hipótese relevante (A+B+C), preencha:
   - O que favorece / O que contradiz / Dados ausentes
   - Exame discriminatório ideal + limitações (falso-negativo, janela, sensibilidade)
   - Achado confirmatório / Achado que exclui
   - Próximo passo
4. BUSCA NOS PDFs: Use trechos acima para sustentar/refutar cada hipótese.
5. REAVALIAÇÃO BAYESIANA: Atualize probabilidades com achados.
5. SAFETY CHECK: Diagnóstico fatal excluído? Condição tempo-dependente? Risco materno/fetal? Anti-ancoragem?
6. CONCLUSÃO: Diagnóstico provisório + nível de incerteza + dados necessários.
7. CONDUTA + FOLLOW-UP.

=== FORMATO DE SAÍDA (OBRIGATÓRIO) ===
Responda EXATAMENTE neste formato Markdown:

## DIFFERENTIAL MATRIX ENGINE — GO

### SÍNDROME/QUEIXA
[descrição]

### CONTEXTO CLÍNICO
[resumo]

### HIPÓTESES GERADAS
**CAMADA A (Principal):**
1. [...] — prob. X%, compat. Y%
**CAMADA B (Relevantes):**
1. [...]
**CAMADA C — NÃO PODEM SER PERDIDOS ⚠️:**
1. [...] — gravidade: crítica, urgência: imediata
**CAMADA D — Mimetizadores:**
• [...]

### MATRIZ DE DISCRIMINAÇÃO
Para cada hipótese relevante (A+B+C):
**Hipótese: [...]
- O que favorece: [...]
- O que contradiz: [...]
- Dados ausentes: [...]
- Exame discriminatório ideal: [...]
- Limitações do exame (falso-negativo, janela, sensibilidade): [...]
- Achado que CONFIRMA: [...]
- Achado que EXCLUI: [...]
- Próximo passo: [...]

### BUSCA NOS PDFs (FEBRASGO/Williams/SOGIMIG)
[Trechos citados com livro + página]

### SAFETY CHECK
☐ Diagnóstico fatal adequadamente excluído?
☐ Condição tempo-dependente abordada?
☐ Risco materno/fetal abordado?
☐ Encaminhamento/internação necessário?
☐ Anti-ancoragem verificada?
⚠️ ALERTAS: [...]

### CONCLUSÃO PROVISÓRIA
**Diagnóstico:** [...]
**Nível de incerteza:** [baixo/moderado/alto]
**Dados necessários para reduzir incerteza:** [...]

### CONDUTA + FOLLOW-UP
1. [...]
2. [...]

REGRAS:
- NÃO invente informação fora dos trechos dos PDFs.
- Se incerteza alta → declare + liste dados necessários.
- SEMPRE cite livro + página.
- Anti-ancoragem: questione "é só X", "exame normal", "β-hCG baixo".
"""

    def generate_initial_hypotheses(self, context: ClinicalContext) -> List[Hypothesis]:
        """Gera hipóteses iniciais baseadas na queixa + contexto."""
        sys_prompt = """Você é especialista em Ginecologia/Obstetrícia.
Gere hipóteses diagnósticas AMPLAS para a queixa/clínica dada.
Organize por SISTEMA (Ginecológico, Obstétrico, GI, Urinário, Vascular, etc.).
Para cada hipótese, estime: probabilidade (0-1), severidade (1-5), urgência (1-5).
Inclua diagnósticos que NÃO PODEM SER PERDIDOS (baixa prob + alta gravidade).
Retorne JSON válido com lista de objetos:
{"diagnosis": str, "probability": float, "severity": int, "urgency": int,
 "system": str, "layer": "A|B|C|D"}"""

        user_prompt = f"""QUEIXA: {context.complaint}
CONTEXTO: idade={context.age}, IG={context.gestational_age_weeks}, G{context.gravida}P{context.para}
Gestante: {context.is_pregnant}
Sinais vitais: {context.vital_signs}
Exame físico: {context.physical_exam}
Fatores de risco: {context.risk_factors}
Labs: {context.labs}
Imagem: {context.imaging}"""

        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=4000,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        hypotheses = []
        for h in data.get("hypotheses", []):
            hypotheses.append(Hypothesis(
                diagnosis=h["diagnosis"],
                probability=h["probability"],
                severity=h["severity"],
                urgency=h["urgency"],
                compatibility=0.5,  # será atualizado
                exclusion_strength=0.5,
                layer=h.get("layer", "B"),
                system=h.get("system", "Outro"),
                favoring=[], against=[], missing=[],
                best_test="", test_limitations="",
                confirmatory_finding="", excluding_finding="",
                next_step=""
            ))
        return hypotheses

    def search_pdf_for_hypothesis(self, hypothesis: Hypothesis, context: ClinicalContext) -> List[Dict]:
        """Busca trechos relevantes para uma hipótese específica."""
        query = f"{context.complaint} {hypothesis.diagnosis} {hypothesis.system}"
        ctxs = self.hybrid_search(query, k=8, max_dist=0.68)
        validated = self.validate_chunks(f"{context.complaint} {hypothesis.diagnosis}", ctxs)
        return validated

    def run_dme(self, context: ClinicalContext) -> str:
        """Pipeline completo DME."""
        # 1. Hipóteses iniciais
        hypotheses = self.generate_initial_hypotheses(context)

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


def parse_clinical_input(text: str) -> ClinicalContext:
    """Parse simples de entrada livre para ClinicalContext.
    Em produção, usar LLM para extrair entidades."""
    ctx = ClinicalContext(complaint=text)
    # Heurísticas simples
    text_lower = text.lower()
    if any(w in text_lower for w in ["gestante", "gestação", "semanas", "ig ", "ig:", "semana"]):
        ctx.is_pregnant = True
        m = re.search(r"(\d+)\s*sem", text_lower)
        if m:
            ctx.gestational_age_weeks = int(m.group(1))
    m = re.search(r"(\d+)\s*anos?", text_lower)
    if m:
        ctx.age = int(m.group(1))
    m = re.search(r"g(\d+)p(\d+)", text_lower)
    if m:
        ctx.gravida, ctx.para = int(m.group(1)), int(m.group(2))
    return ctx


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