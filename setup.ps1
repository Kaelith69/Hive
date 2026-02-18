# 🚀 Quick Start Script for Hive Training
# Run this PowerShell script to set up your training environment quickly

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "🚀 Hive Personality LLM Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "1️⃣  Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "   ❌ Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Step 2: Check CUDA
Write-Host ""
Write-Host "2️⃣  Checking CUDA..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda if torch.cuda.is_available() else None}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ $cudaCheck" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  PyTorch not installed yet - will install in next step" -ForegroundColor Yellow
}

# Step 3: Validate dataset
Write-Host ""
Write-Host "3️⃣  Validating dataset..." -ForegroundColor Yellow
python validate_dataset.py --input dataset/final/personality.jsonl
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Dataset validation complete" -ForegroundColor Green
} else {
    Write-Host "   ❌ Dataset validation failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Installation prompt
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "📦 Next Steps:" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To install training dependencies, run:" -ForegroundColor White
Write-Host ""
Write-Host "# 1. Create virtual environment (recommended)" -ForegroundColor Gray
Write-Host "python -m venv venv" -ForegroundColor Yellow
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "# 2. Install PyTorch with CUDA support" -ForegroundColor Gray
Write-Host 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121' -ForegroundColor Yellow
Write-Host ""
Write-Host "# 3. Install Unsloth (fast training library)" -ForegroundColor Gray
Write-Host 'pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"' -ForegroundColor Yellow
Write-Host ""
Write-Host "# 4. Install other dependencies" -ForegroundColor Gray
Write-Host "pip install transformers datasets accelerate peft trl bitsandbytes pyyaml" -ForegroundColor Yellow
Write-Host ""
Write-Host "# 5. Start training!" -ForegroundColor Gray
Write-Host "python train.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "For detailed instructions, see TRAINING_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
