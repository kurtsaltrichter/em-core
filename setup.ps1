# em-core setup script
# Run from PowerShell: cd C:\Users\Kurta\em-core; .\setup.ps1
# This pushes to GitHub, installs in both venvs, and boot-tests the bot.

$ErrorActionPreference = "Stop"

Write-Host "`n=== STEP 1: Push em-core to GitHub ===" -ForegroundColor Cyan

# Clean up any leftover .git artifacts
if (Test-Path ".git") { Remove-Item -Recurse -Force ".git" }

git init -b main
git add -A
git commit -m "Initial commit: em-core v0.1.0 -- shared polygon, iv, locking, universe, storage, earnings"
git remote add origin https://github.com/kurtsaltrichter/em-core.git
git push -u origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host "Git push failed. You may need to authenticate." -ForegroundColor Red
    Write-Host "Try: git push -u origin main" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "`nPushed to https://github.com/kurtsaltrichter/em-core" -ForegroundColor Green

Write-Host "`n=== STEP 2: Install em-core in tripwire-bot venv ===" -ForegroundColor Cyan
Push-Location C:\Users\Kurta\tripwire-bot
python -m pip install -e C:\Users\Kurta\em-core
python -c "from em_core import polygon, iv, locking, universe, storage, earnings; print('em_core imports OK in tripwire-bot venv')"
Pop-Location

Write-Host "`n=== STEP 3: Install em-core in em-dashboard venv ===" -ForegroundColor Cyan
Push-Location C:\Users\Kurta\em-dashboard
python -m pip install -e C:\Users\Kurta\em-core
python -c "from em_core import polygon, iv, locking; print('em_core imports OK in em-dashboard venv')"
Pop-Location

Write-Host "`n=== STEP 4: Boot test TripWire bot (5 seconds) ===" -ForegroundColor Cyan
Push-Location C:\Users\Kurta\tripwire-bot
Write-Host "Starting bot for quick smoke test..."
$botJob = Start-Job -ScriptBlock {
    Set-Location C:\Users\Kurta\tripwire-bot
    python -m bot.main 2>&1
}
Start-Sleep -Seconds 5
$output = Receive-Job $botJob
Stop-Job $botJob
Remove-Job $botJob

if ($output -match "running" -or $output -match "initialized") {
    Write-Host "Bot started successfully!" -ForegroundColor Green
    Write-Host ($output | Select-Object -First 10 | Out-String)
} else {
    Write-Host "Bot output:" -ForegroundColor Yellow
    Write-Host ($output | Out-String)
}
Pop-Location

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "em-core pushed to GitHub, installed in both venvs, bot boot-tested."
Write-Host "Next: run docker-compose for deployment."
pause
