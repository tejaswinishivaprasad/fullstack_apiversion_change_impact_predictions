# AI Core (FastAPI)

Minimal, runnable AI core exposing `/report` that returns a structured backend vs frontend impact report.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.server:app --reload --port 8000
# http://localhost:8000/docs
```

## API
`GET /report?old=...&new=...` -> JSON
