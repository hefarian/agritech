$ErrorActionPreference = "Stop"

docker compose up -d --build api frontend
Write-Host "Frontend demarree. Port pilote par .env" -ForegroundColor Green
