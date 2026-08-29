# Pride & Prejudice Translation Pipeline - Automated Setup (Windows PowerShell)
#
# Usage:
#   cd book-translate
#   .\setup.ps1
#
# This script:
#   1. Verifies Python is installed
#   2. Creates .venv if it doesn't exist (reuses it if it does)
#   3. Activates .venv for this script's session
#   4. Detects NVIDIA GPU + CUDA version (falls back to CPU-only torch)
#   5. Installs all dependencies from requirements.txt
#   6. Creates .env from .env.example if missing
#   7. Checks whether Ollama is installed / running
#   8. Prints next steps

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Write-Warn($msg) {
    Write-Host "[warn] $msg" -ForegroundColor Yellow
}

function Write-Ok($msg) {
    Write-Host "[ok] $msg" -ForegroundColor Green
}

# --- 1. Check Python ---
Write-Step "Checking Python installation"
try {
    $pyVersion = & python --version 2>&1
    Write-Ok "Found $pyVersion"
} catch {
    Write-Host "[error] Python not found in PATH. Install Python 3.9+ from python.org and retry." -ForegroundColor Red
    exit 1
}

# --- 2. Create venv if missing ---
Write-Step "Setting up virtual environment (.venv)"
if (Test-Path ".venv") {
    Write-Ok ".venv already exists - reusing it"
} else {
    python -m venv .venv
    Write-Ok "Created .venv"
}

# --- 3. Activate venv for this script session ---
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "[error] Could not find $activateScript - venv creation may have failed." -ForegroundColor Red
    exit 1
}
. $activateScript
Write-Ok "Activated .venv for this session"

# --- 4. Detect NVIDIA GPU / CUDA ---
Write-Step "Detecting GPU / CUDA"
$cudaIndex = $null
try {
    $nvidiaOutput = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $cudaLine = $nvidiaOutput | Select-String "CUDA Version:"
        if ($cudaLine) {
            $cudaVersionStr = ($cudaLine -split "CUDA Version:\s*")[1].Trim().Split(" ")[0]
            Write-Ok "NVIDIA GPU detected - driver supports CUDA $cudaVersionStr"
            $cudaMajorMinor = [double]($cudaVersionStr -replace '^(\d+\.\d+).*', '$1')
            if ($cudaMajorMinor -ge 12.4) {
                $cudaIndex = "cu124"
            } elseif ($cudaMajorMinor -ge 12.0) {
                $cudaIndex = "cu121"
            } else {
                $cudaIndex = "cu118"
            }
            Write-Ok "Will install torch with $cudaIndex wheels"
        } else {
            Write-Warn "nvidia-smi ran but couldn't parse CUDA version - defaulting to CPU-only torch"
        }
    }
} catch {
    Write-Warn "No NVIDIA GPU detected (nvidia-smi not found) - installing CPU-only torch"
}

# --- 5. Install dependencies ---
Write-Step "Installing Python dependencies"
python -m pip install --upgrade pip | Out-Null

if ($cudaIndex) {
    # requirements.txt already pins --extra-index-url cu121 by default.
    # If detected version differs, install torch separately with the right index first.
    if ($cudaIndex -ne "cu121") {
        Write-Step "Installing torch for $cudaIndex (overriding requirements.txt default)"
        pip install torch==2.5.1 --index-url "https://download.pytorch.org/whl/$cudaIndex"
    }
    pip install -r requirements.txt
} else {
    Write-Step "Installing CPU-only torch"
    pip install torch==2.5.1
    Write-Step "Installing remaining dependencies (skipping torch line from requirements.txt)"
    Get-Content requirements.txt | Where-Object { $_ -notmatch "^torch==" -and $_ -notmatch "^--extra-index-url" } | Set-Content ".requirements.tmp.txt"
    pip install -r ".requirements.tmp.txt"
    Remove-Item ".requirements.tmp.txt"
}
Write-Ok "Dependencies installed"

# --- 6. Set up .env ---
Write-Step "Checking .env configuration"
if (Test-Path ".env") {
    Write-Ok ".env already exists - leaving it as-is"
} elseif (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Warn "Created .env from .env.example - edit it and add your GROQ_API_KEY before running the pipeline"
} else {
    Write-Warn "No .env or .env.example found - you'll need to create .env manually with GROQ_API_KEY and model names"
}

# --- 7. Check Ollama ---
Write-Step "Checking Ollama"
try {
    $ollamaVersion = & ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Ollama installed: $ollamaVersion"
        try {
            Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null
            Write-Ok "Ollama server is running"
        } catch {
            Write-Warn "Ollama is installed but not running - start it with: ollama serve"
        }
    }
} catch {
    Write-Warn "Ollama not found - install from https://ollama.ai if you plan to use local models (backtranslate/QA/polish)"
}

# --- 8. Summary ---
Write-Step "Setup complete"
Write-Host @"

Next steps:
  1. Edit .env and set GROQ_API_KEY (and verify model names are current - Groq
     rotates model availability, check https://console.groq.com/docs/models)
  2. If using local models, make sure Ollama is running:  ollama serve
     and pull the models referenced in .env, e.g.:
       ollama pull llama3.1:8b
       ollama pull mistral-nemo:12b
  3. Activate the venv in new terminals with:
       .\.venv\Scripts\Activate.ps1
  4. Run the pipeline:
       python scripts\run_pipeline.py --book "Pride and Prejudice - Jane Austen" --chapters 3 --polish

See SETUP.md for troubleshooting.
"@
