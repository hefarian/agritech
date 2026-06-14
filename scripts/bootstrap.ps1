$ErrorActionPreference = "Stop"

# Bootstrap Docker-only workflow by building all images once.
docker compose build
