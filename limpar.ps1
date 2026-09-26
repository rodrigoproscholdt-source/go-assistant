$ErrorActionPreference = "SilentlyContinue"
Set-Location D:\GO_assit
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host ("=" * 64) -ForegroundColor White
Write-Host "   LIMPEZA DA PASTA  D:\GO_assit" -ForegroundColor White
Write-Host ("=" * 64) -ForegroundColor White
Write-Host ""

$scratch = @(
    # diagnostico desta sessao
    "test_fts.py", "test_scale.py", "test_api.py", "diag_db.py",
    # testes antigos (contem a chave Groq exposta)
    "test_pipeline.py", "test_structured.py", "test_endo.py",
    "test_endo2.py", "test_endo3.py", "test_bm25.py", "test_q.py",
    # rascunhos .txt
    "hybrid_test.txt", "endo1.txt", "booktitles.txt", "ctx_dump.txt",
    "ctx_dump2.txt", "coldet.txt", "coldet2.txt", "coldet3.txt",
    "col_test.txt", "quality_test.txt", "phrase_hits.txt", "mem_hits.txt",
    "cov3.txt", "cov4.txt", "cov5.txt", "coverage.txt", "coverage2.txt",
    "pipeline_log.txt", "pipeline_log2.txt",
    # rascunhos .md
    "endo_v2.md", "endo_estruturado.md", "teste_resumo.md",
    "endo_app_sim.md", "prenatal_v2.md", "pipeline_test.md",
    # SKILL.md solto na raiz (a real esta em .opencode\skill\)
    "SKILL.md",
    # scripts superados pelo app_streamlit.py + dme_engine.py
    "go_groq_rag.py", "go_assistant.py",
    # requirements duplicados
    "requirements_groq.txt", "requirements_streamlit.txt"
)

$removidos = 0
foreach ($f in $scratch) {
    if (Test-Path $f) {
        Remove-Item $f -Force
        Write-Host ("  removido : {0}" -f $f) -ForegroundColor DarkGray
        $removidos++
    }
}

if (Test-Path __pycache__) {
    Remove-Item __pycache__ -Recurse -Force
    Write-Host "  removido : __pycache__" -ForegroundColor DarkGray
    $removidos++
}

Write-Host ""
Write-Host "  -- limpando do indice do git --" -ForegroundColor Cyan
foreach ($f in @("index_err.txt", "go_assistant.py", "go_groq_rag.py",
                 "requirements_groq.txt", "requirements_streamlit.txt")) {
    if (git ls-files --error-unmatch $f 2>$null) {
        git rm --cached -q $f 2>$null
        Write-Host ("  git rm --cached : {0}" -f $f) -ForegroundColor Cyan
    }
}
$pdfs = @(Get-ChildItem *.pdf)
foreach ($p in $pdfs) {
    if ($p.Length -ge 99MB) {
        git rm --cached -q -- $p.Name 2>$null
        Write-Host ("  git rm --cached : {0}  ({1:N1} MB - acima do limite GitHub)" -f $p.Name, ($p.Length / 1MB)) -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host ("  total de arquivos locais removidos: {0}" -f $removidos) -ForegroundColor Green
Write-Host ""
Write-Host ("=" * 64) -ForegroundColor White
Write-Host "  CONTEUDO DA PASTA" -ForegroundColor White
Write-Host ("=" * 64) -ForegroundColor White
Get-ChildItem -Force | Sort-Object PSIsContainer, Name | ForEach-Object {
    if ($_.PSIsContainer) {
        $mb = (Get-ChildItem $_.FullName -Recurse -File -EA SilentlyContinue |
               Measure-Object Length -Sum).Sum / 1MB
        Write-Host ("   [dir] {0,-46} {1,8:N1} MB" -f $_.Name, $mb) -ForegroundColor DarkCyan
    } else {
        Write-Host ("         {0,-46} {1,8:N2} MB" -f $_.Name, ($_.Length / 1MB)) -ForegroundColor Gray
    }
}
Write-Host ""
