$ErrorActionPreference = "Stop"

docker compose up -d --build api
Write-Host "API demarree. Port pilote par .env" -ForegroundColor Green
