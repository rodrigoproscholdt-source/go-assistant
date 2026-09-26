$ErrorActionPreference = "SilentlyContinue"
Set-Location D:\GO_assit
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Line($c = "-") { Write-Host ("-" * 64) -ForegroundColor DarkGray }
function Head($t) { Line; Write-Host $t -ForegroundColor Cyan }
function Ok($t)   { Write-Host $t -ForegroundColor Green }
function Warn($t) { Write-Host $t -ForegroundColor Yellow }
function Bad($t)  { Write-Host $t -ForegroundColor Red }
function Info($t) { Write-Host $t -ForegroundColor Gray }

Write-Host ""
Write-Host ("=" * 64) -ForegroundColor White
Write-Host "   GO_ASSIST  |  STATUS DO SISTEMA" -ForegroundColor White
Write-Host ("   {0} - {1}" -f (Get-Date -Format "dd/MM/yyyy"), (Get-Date -Format "HH:mm:ss")) -ForegroundColor White
Write-Host ("=" * 64) -ForegroundColor White
Write-Host ""

# ---------- 1. PROCESSO ----------
Head "[1/7] PROCESSO DE INDEXACAO"
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*index_local*" }
if ($proc) {
    foreach ($p in $proc) {
        Write-Host ("  ATIVO  PID {0}   inicio {1}" -f $p.ProcessId, $p.CreationDate) -ForegroundColor Green
    }
    $secs = ((Get-Date) - $proc.CreationDate).TotalMinutes
    if ($secs -gt 0) { Info ("  tempo decorrido: {0:N0} min" -f $secs) }
} else {
    Warn "  PARADO - nenhum index_local.py em execucao"
}

# ---------- 2. LOG ----------
Head "[2/7] ULTIMAS LINHAS DO LOG"
if (Test-Path index_log.txt) {
    Get-Content index_log.txt -Tail 6 -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }
    $concl = (Select-String -Path index_log.txt -Pattern "Concluido" -Encoding UTF8 | Measure-Object).Count
    if ($concl -gt 0) {
        Info ("  livros concluidos: {0} de 3" -f $concl)
        Select-String -Path index_log.txt -Pattern "Concluido" -Encoding UTF8 |
            ForEach-Object { Write-Host ("    " + $_.Line) -ForegroundColor DarkYellow }
    }
} else { Warn "  index_log.txt nao encontrado" }

# ---------- 3. CHUNKS ----------
Head "[3/7] CHUNKS INDEXADOS"
$py = (Get-Command python -EA SilentlyContinue).Source
if ($py) {
    $code = @'
import chromadb, collections
c = chromadb.PersistentClient(path="chroma_db").get_collection("go_books")
print("TOTAL:", c.count(), "chunks")
m = collections.Counter(x["book"] for x in c.get(include=["metadatas"])["metadatas"])
for k, v in sorted(m.items()):
    print("   -", k, ":", v, "chunks")
'@
    $code | Out-File -FilePath "$env:TEMP\_go_status.py" -Encoding UTF8
    $out = & python -X utf8 "$env:TEMP\_go_status.py" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $out | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
    } else { Warn "  collection vazia ou ilegivel" }
} else { Warn "  python nao encontrado no PATH" }

# ---------- 4. TAMANHO ----------
Head "[4/7] TAMANHO  (limite GitHub = 100 MB por arquivo)"
$files = Get-ChildItem chroma_db -Recurse -File -EA SilentlyContinue
if ($files) {
    $total = ($files | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("  TOTAL chroma_db : {0:N1} MB" -f $total) -ForegroundColor White
    $files | Sort-Object Length -Desc | Select-Object -First 5 | ForEach-Object {
        Write-Host ("   {0,-40} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) -ForegroundColor DarkGray
    }
    $big = $files | Where-Object { $_.Length -ge 99MB }
    if ($big) {
        foreach ($b in $big) { Bad ("  BLOQUEADO: {0} tem {1:N1} MB" -f $b.Name, ($b.Length / 1MB)) }
    }
    if ($total -lt 99) { Ok "  [OK] total abaixo de 99 MB" }
    else { Warn "  [ATENCAO] acima de 99 MB - push direto sera BLOQUEADO" }
} else { Warn "  chroma_db nao existe" }

# ---------- 5. GIT ----------
Head "[5/7] GITHUB"
$last = git log -1 --pretty=format:"%h  %s" 2>$null
if ($last) { Write-Host "  HEAD local : $last" -ForegroundColor White }
$st = @(git status --porcelain 2>$null)
Write-Host ("  arquivos alterados/nao rastreados: {0}" -f $st.Count) -ForegroundColor White
$interest = $st | Where-Object { $_ -match "chroma_db|index_local|app_streamlit|dme_engine|SKILL|\.pdf" }
if ($interest) {
    Write-Host "  itens de interesse:" -ForegroundColor Yellow
    $interest | Select-Object -First 12 | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    if ($interest.Count -gt 12) { Info ("    ... e mais {0}" -f ($interest.Count - 12)) }
} else { Ok "  nada pendente nos arquivos de interesse" }

# ---------- 6. CONFIG ----------
Head "[6/7] CONFIGURACAO"
if ($env:GROQ_API_KEY) { Ok "  GROQ_API_KEY : definida no ambiente" }
else { Warn "  GROQ_API_KEY : AUSENTE no ambiente local (cloud usa Streamlit Secret)" }
Select-String -Path index_local.py -Pattern "^(CHUNK_WORDS|OVERLAP_WORDS)\s*=" |
    ForEach-Object { Write-Host ("  index_local.py : " + $_.Line.Trim()) -ForegroundColor White }
Select-String -Path app_streamlit.py,dme_engine.py -Pattern "GROQ_MODEL\s*=" | ForEach-Object {
    $m = ($_.Line -replace '.*=\s*', '')
    Write-Host ("  {0} : GROQ_MODEL = {1}" -f $_.Filename, $m) -ForegroundColor White
}
$env:GROQ_API_KEY = $null

# ---------- 7. PDFs ----------
Head "[7/7] PDFs DA BASE"
Get-ChildItem *.pdf | ForEach-Object {
    $mb = $_.Length / 1MB
    $col = if ($mb -ge 99) { "Red" } else { "Gray" }
    Write-Host ("  {0,-52} {1,7:N1} MB" -f $_.Name, $mb) -ForegroundColor $col
}
Write-Host ""
Write-Host ("=" * 64) -ForegroundColor White
Write-Host "  Log completo : type index_log.txt" -ForegroundColor Gray
Write-Host "  Erros         : type index_err.txt" -ForegroundColor Gray
Write-Host ("=" * 64) -ForegroundColor White
Write-Host ""
