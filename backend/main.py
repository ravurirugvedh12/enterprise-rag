"""
main.py — FastAPI server for the Enterprise RAG application
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise RAG API",
    description="Internal knowledge base search powered by LangChain + OpenAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance — initialised lazily so missing keys don't crash startup
_engine: Optional[RAGEngine] = None

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    k: int = 5


class SettingsRequest(BaseModel):
    api_key: Optional[str] = ""
    model: Optional[str] = "qwen2:0.5b"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "message": "Enterprise RAG API is running"}


@app.get("/stats")
def stats():
    try:
        engine = get_engine()
        return engine.stats()
    except HTTPException:
        return {"total_documents": 0, "total_chunks": 0, "model": "—", "embedding_model": "—"}


@app.post("/settings")
def update_settings(req: SettingsRequest):
    """Update model settings (for Ollama, api_key is optional)."""
    global _engine
    if req.model:
        os.environ["OLLAMA_MODEL"] = req.model
    try:
        _engine = RAGEngine()
        return {"status": "ok", "message": "Settings updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the knowledge base."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    engine = get_engine()

    # Save to a temp file so loaders can read it from disk
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        record = engine.ingest_file(tmp_path, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return {
        "status":   "success",
        "message":  f"'{file.filename}' ingested successfully.",
        "document": record,
    }


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    try:
        engine = get_engine()
        return {"documents": engine.list_documents()}
    except HTTPException:
        return {"documents": []}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Remove a document and all its chunks from the knowledge base."""
    engine = get_engine()
    deleted = engine.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "success", "message": f"Document '{doc_id}' removed."}


@app.post("/query")
def query(req: QueryRequest):
    """Ask a question against the knowledge base."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    engine = get_engine()
    try:
        result = engine.query(req.question, k=req.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
