$ErrorActionPreference = "Stop"

# Rebuild consolidated dataset then train model artifacts in containers.
docker compose run --rm preprocess
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

docker compose run --rm train
