$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
