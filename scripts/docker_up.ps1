$ErrorActionPreference = "Stop"

docker compose up -d --build api frontend mlflow
Write-Host "Services demarres. Les ports sont lus depuis le fichier .env" -ForegroundColor Green
