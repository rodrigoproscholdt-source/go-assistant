"""Testes do extrator de contexto clinico.

Foco: nao confundir 'falta de menstruacao ha 4 semanas' (amenorreia) com
'gestante de 4 semanas' (IG). Erro classico e perigoso.
"""
import sys

from dme_engine import ClinicalContext, _pregnancy_block, parse_clinical_input

# (entrada, gestante, status, IG, amen_sem, amen_dias, idade)
CASOS = [
    ("mulher procura atendimento medico com relato de falta de menstruacao ha 4 semanas?",
     False, "incerta", None, 4, None, None),
    ("gestante de 20 semanas com dor abdominal",
     True, "", 20, None, None, None),
    ("G2P1 ha 30 semanas, PA 160/100, cefaleia",
     True, "", 30, None, None, None),
    ("IG de 12 semanas sem sangramento",
     True, "", 12, None, None, None),
    ("mulher 28 anos com dor pelvica",
     False, "", None, None, None, 28),
    ("amenorreia ha 3 meses",
     False, "incerta", None, None, 90, None),
    ("nao menstrua ha 60 dias",
     False, "incerta", None, None, 60, None),
    ("gestante com 8 semanas",
     True, "", 8, None, None, None),
    ("mulher 35 anos, G3P2, 39 semanas de gestacao, sangramento vaginal",
     True, "", 39, None, None, 35),
    ("paciente com atraso menstrual de 6 semanas e dor em fossa iliaca",
     False, "incerta", None, 6, None, None),
]

falhas = 0
print(f"{'entrada':<62} {'gest':<5} {'status':<9} {'IG':<4} {'am.sem':<7} {'am.dias'}")
print("-" * 100)
for texto, gest, status, ig, ams, amd, idade in CASOS:
    c = parse_clinical_input(texto)
    erros = []
    if c.is_pregnant != gest:
        erros.append(f"gestante={c.is_pregnant} esperado {gest}")
    if c.pregnancy_status != status:
        erros.append(f"status={c.pregnancy_status!r} esperado {status!r}")
    if c.gestational_age_weeks != ig:
        erros.append(f"IG={c.gestational_age_weeks} esperado {ig}")
    if c.amenorrhea_weeks != ams:
        erros.append(f"amen_sem={c.amenorrhea_weeks} esperado {ams}")
    if c.amenorrhea_days != amd:
        erros.append(f"amen_dias={c.amenorrhea_days} esperado {amd}")
    if c.age != idade:
        erros.append(f"idade={c.age} esperado {idade}")

    marca = "OK  " if not erros else "ERRO"
    if erros:
        falhas += 1
    print(f"{marca} {texto[:60]:<60} {str(c.is_pregnant):<5} "
          f"{c.pregnancy_status:<9} {str(c.gestational_age_weeks or '-'):<4} "
          f"{str(c.amenorrhea_weeks or '-'):<7} {c.amenorrhea_days or '-'}")
    for e in erros:
        print(f"       -> {e}")
    print(f"       bloco: {_pregnancy_block(c)}")

print()
if falhas:
    print(f"FALHOU em {falhas}/{len(CASOS)} caso(s)")
    sys.exit(1)
print(f"CONTEXTO: {len(CASOS)}/{len(CASOS)} casos corretos")
