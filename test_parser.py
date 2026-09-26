"""Testes do parser de hipoteses: nenhum formato de resposta pode quebrar o app."""
import sys

from dme_engine import DMEEngine

casos = [
    ("dict com hypotheses", '{"hypotheses":[{"diagnosis":"Gravidez ectopica",'
     '"probability":0.4,"severity":5,"urgency":5,"system":"Obstetrico","layer":"C"}]}'),
    ("lista pura", '[{"diagnosis":"Amenorreia","probability":0.5,'
     '"severity":3,"urgency":3,"system":"Ginecologico","layer":"A"}]'),
    ("lista vazia (o bug do cloud)", '[]'),
    ("cercas de markdown", '```json\n{"hypotheses":[{"diagnosis":"Anemia",'
     '"probability":0.3,"severity":2,"urgency":2}]}\n```'),
    ("texto antes do json", 'Aqui estao as hipoteses:\n{"hypotheses":'
     '[{"diagnosis":"Amenorreia","probability":0.6,"severity":3,"urgency":3}]}'),
    ("chamada em texto", '[{"diagnosis":"Emenorragia por Coincidencia",'
     '"probability":0.7,"severity":2,"urgency":2,"system":"Ginecologico",'
     '"layer":"A","favoring":["4 meses de amenorreia"],'
     '"against":[],"missing":["beta-hCG"],'
     '"best_test":"beta-hCG quantitativo",'
     '"test_limitations":"falso-negativo se teste tardio",'
     '"confirmatory_finding":"beta-hCG positivo",'
     '"excluding_finding":"beta-hCG negativo e utero vazio",'
     '"next_step":"beta-hCG e USG"}]}'),
    ("strings soltas", '["Gravidez ectopica","Abortamento"]'),
    ("campos faltando", '{"hypotheses":[{"diagnosis":"Teste"}]}'),
    ("campos com texto errado", '{"hypotheses":[{"diagnosis":"X","probability":"alto",'
     '"severity":null,"urgency":"3","layer":9}]}'),
    ("vazio", ''),
    ("lixo", 'resposta sem json'),
]

falhas = 0
for nome, raw in casos:
    try:
        h = DMEEngine._parse_hypotheses(raw)
        nomes = [x.diagnosis for x in h]
        status = "OK " if raw.strip() == "" or raw == "resposta sem json" or h or raw == "[]" else "VAZIO"
        print(f"  [{status:<5}] {nome:<32} -> {len(h)} hipotese(s) {nomes[:3]}")
        if status == "VAZIO":
            falhas += 1
    except Exception as e:
        print(f"  [ERRO ] {nome:<32} -> {type(e).__name__}: {e}")
        falhas += 1

print()
if falhas:
    print(f"FALHOU em {falhas} caso(s)")
    sys.exit(1)
print("PARSER: todos os formatos tolerados")
