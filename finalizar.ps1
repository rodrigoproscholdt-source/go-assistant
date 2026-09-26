$ErrorActionPreference = "Continue"
Set-Location D:\GO_assit
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$LOG = "D:\GO_assit\finalizar_log.txt"

function Say($msg, $color = "Gray") {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}
function Gate($nome, $ok, $detalhe) {
    if ($ok) { Say "  GATE OK   $nome  $detalhe" "Green" }
    else     { Say "  GATE FAIL $nome  $detalhe" "Red" }
    return $ok
}

Say "==== INICIO DA FINALIZACAO AUTOMATICA ====" "White"

# ---------------------------------------------------------------- 1. INDEXACAO
Say "[1/6] aguardando index_local.py terminar..."
$deadline = (Get-Date).AddMinutes(90)
$done = $false
while ((Get-Date) -lt $deadline) {
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like "*index_local*" }
    if (-not $proc) {
        Start-Sleep 3
        if (Test-Path index_log.txt) {
            if (Select-String -Path index_log.txt -Pattern "Indexacao concluida com sucesso" -EA SilentlyContinue) {
                $done = $true
            } else {
                Say "index_local.py encerrou SEM a mensagem de sucesso - abortando" "Red"
                Say "veja type index_err.txt" "Yellow"
                break
            }
        }
        break
    }
    $last = (Get-Content index_log.txt -Tail 1 -EA SilentlyContinue)
    Say ("  indexando... {0}" -f $last) "DarkGray"
    Start-Sleep 60
}

if (-not $done) { Say "ABORTADO: indexacao nao concluiu" "Red"; exit 1 }

$concl = @(Select-String -Path index_log.txt -Pattern "Concluido:" | ForEach-Object { $_.Line })
Say "indexacao concluida. Livros processados: $($concl.Count)" "Green"
$concl | ForEach-Object { Say "   $_" "DarkGray" }
if (-not (Gate "3 livros indexados" ($concl.Count -eq 3) "($($concl.Count)/3)")) { exit 1 }

# ---------------------------------------------------------------- 2. SPLIT
Say "[2/6] dividindo a base por livro (split_db.py)..."
$out = & python -X utf8 split_db.py 2>&1
$out | ForEach-Object { Say "   $_" "DarkGray" }
$splitOk = $LASTEXITCODE -eq 0
if (-not (Gate "split_db.py executado" $splitOk "exit=$LASTEXITCODE")) { exit 1 }

# ---------------------------------------------------------------- 3. TAMANHO
Say "[3/6] verificando limite de 99 MB por arquivo..."
$targets = @("chroma_db_williams", "chroma_db_sogimig", "chroma_db_febrasgo")
$sizeOk = $true
foreach ($t in $targets) {
    if (-not (Test-Path $t)) { Say "   FALTA $t" "Red"; $sizeOk = $false; continue }
    $mb = (Get-ChildItem $t -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    $big = @(Get-ChildItem $t -Recurse -File | Where-Object { $_.Length -ge 99MB })
    if ($mb -ge 99 -or $big.Count -gt 0) { $sizeOk = $false }
    if ($mb -lt 99) {
        Say ("   {0,-22} {1,7:N2} MB [OK]" -f $t, $mb) "Green"
    } else {
        Say ("   {0,-22} {1,7:N2} MB [ACIMA]" -f $t, $mb) "Red"
    }
}
if (-not (Gate "todas as bases < 99 MB" $sizeOk "")) {
    Say "ABORTADO: base acima do limite do GitHub - push nao realizado" "Red"
    exit 1
}

# ---------------------------------------------------------------- 4. SMOKE TEST
Say "[4/6] smoke test da recuperacao (3 livros, sem LLM)..."
$st = & python -X utf8 smoke_test.py 2>&1
$st | ForEach-Object { Say "   $_" "DarkGray" }
if (-not (Gate "smoke test" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE")) { exit 1 }

# ---------------------------------------------------------------- 5. GIT
Say "[5/6] commitando as bases..."
git rm --cached -q -r chroma_db 2>$null
git add -A chroma_db_williams chroma_db_sogimig chroma_db_febrasgo 2>$null
git add dme_engine.py app_streamlit.py smoke_test.py split_db.py 2>$null
git add -A smoke_test.py finalizar.ps1 2>$null
$staged = @(git diff --cached --name-only)
Say "  arquivos no staged: $($staged.Count)" "DarkGray"
$bigStaged = @()
foreach ($f in $staged) {
    $p = Join-Path (Get-Location) $f
    if (Test-Path -LiteralPath $p) {
        $item = Get-Item -LiteralPath $p
        if (-not $item.PSIsContainer) {
            $mb = $item.Length / 1MB
            if ($mb -ge 99) { $bigStaged += ("{0} ({1:N1} MB)" -f $f, $mb) }
        }
    }
}
if ($bigStaged.Count -gt 0) {
    Say "  ARQUIVOS ACIMA DE 99 MB NO STAGED:" "Red"
    $bigStaged | ForEach-Object { Say "     $_" "Red" }
    Say "ABORTADO: push nao realizado" "Red"
    exit 1
}
git commit -q -m "Base de 3 livros dividida por tratado + smoke test

- chroma_db_williams / chroma_db_sogimig / chroma_db_febrasgo
  (cada chroma.sqlite3 abaixo do limite de 100 MB do GitHub)
- base monolitica chroma_db removida do indice do git
- DMEEngine multi-bases com BM25 global e RRF por shard
- smoke_test.py valida a recuperacao nos 3 livros sem chamar o LLM"
if ($LASTEXITCODE -ne 0) { Say "  (nada a commitar ou commit falhou)" "Yellow" }

# ---------------------------------------------------------------- 6. PUSH
Say "[6/6] enviando para o GitHub..."
$push = git push -u origin main 2>&1
$push | ForEach-Object { Say "   $_" "DarkGray" }
if ($LASTEXITCODE -eq 0) {
    Say "==== CONCLUIDO COM SUCESSO - app sera reimplantado pelo Streamlit Cloud ====" "Green"
} else {
    Say "PUSH FALHOU - ver output acima" "Red"
    exit 1
}
