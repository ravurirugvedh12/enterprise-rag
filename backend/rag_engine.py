"""
rag_engine.py — LangChain RAG pipeline using Ollama (local, free, no API key)
Handles document ingestion, embedding, vector storage, and retrieval-augmented generation.
"""

import os
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_classic.schema import Document

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "qwen2:0.5b")
OLLAMA_EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR    = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE            = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP         = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K", 5))

# Metadata sidecar — persists document info across restarts
METADATA_FILE = Path(CHROMA_PERSIST_DIR) / "doc_metadata.json"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent enterprise knowledge assistant. Use ONLY the provided context to answer the question accurately and concisely.

If the answer is not contained in the context, say "I don't have enough information in the knowledge base to answer this question."

Always cite which document(s) your answer is based on.

Context:
{context}

Question: {question}

Answer:""",
)


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, api_key: Optional[str] = None):
        # api_key is ignored for Ollama (kept for API compatibility)
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name="enterprise_kb",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        self._doc_meta: Dict[str, Dict] = self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _load_metadata(self) -> Dict[str, Dict]:
        if METADATA_FILE.exists():
            try:
                return json.loads(METADATA_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_metadata(self):
        METADATA_FILE.write_text(json.dumps(self._doc_meta, indent=2))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def _load_file(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf":  PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt":  TextLoader,
            ".md":   UnstructuredMarkdownLoader,
        }
        loader_cls = loaders.get(ext)
        if loader_cls is None:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_cls(file_path).load()

    def ingest_file(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        docs = self._load_file(file_path)

        for doc in docs:
            doc.metadata["source_file"] = original_filename
            doc.metadata["doc_id"]      = str(uuid.uuid4())[:8]

        chunks = self.text_splitter.split_documents(docs)
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        self.vectorstore.add_documents(chunks, ids=chunk_ids)

        doc_record = {
            "id":         str(uuid.uuid4()),
            "filename":   original_filename,
            "chunks":     len(chunks),
            "pages":      len(docs),
            "chunk_ids":  chunk_ids,
        }
        self._doc_meta[doc_record["id"]] = doc_record
        self._save_metadata()
        return doc_record

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> bool:
        record = self._doc_meta.get(doc_id)
        if not record:
            return False
        ids = record.get("chunk_ids", [])
        if ids:
            self.vectorstore.delete(ids=ids)
        del self._doc_meta[doc_id]
        self._save_metadata()
        return True

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict]:
        return list(self._doc_meta.values())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str, k: int = RETRIEVAL_K) -> Dict[str, Any]:
        import re as _re

        # Auto-detect specific document mentioned in question
        search_filter = None
        all_filenames = [r["filename"] for r in self._doc_meta.values()]
        for filename in all_filenames:
            name_no_ext = filename.rsplit(".", 1)[0].lower()
            if name_no_ext in question.lower() or filename.lower() in question.lower():
                search_filter = {"source_file": filename}
                break

        search_kwargs = {"k": k}
        if search_filter:
            search_kwargs["filter"] = search_filter

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        # Collect sources
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", None)
            key = f"{src}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file":    src,
                    "page":    page,
                    "excerpt": doc.page_content[:300].strip(),
                })

        # Auto-format as numbered list if user asked for bullet/numbered points
        wants_bullets = bool(_re.search(
            r'(bullet|numbered|points?|list|steps?)\b', question, _re.IGNORECASE
        ))
        already_list = bool(_re.search(r'^\s*\d+\.', answer, _re.MULTILINE))

        if wants_bullets and not already_list:
            sentences = _re.split(r'(?<=[.!?])\s+', answer.strip())
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            num_match = _re.search(r'(\d+)\s*(bullet|numbered|points?|list)', question, _re.IGNORECASE)
            limit = int(num_match.group(1)) if num_match else len(sentences)
            sentences = sentences[:limit]
            answer = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

        return {
            "answer":  answer,
            "sources": sources,
        }


    # ------------------------------------------------------------------
    # Health / stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        total_chunks = self.vectorstore._collection.count()
        return {
            "total_documents": len(self._doc_meta),
            "total_chunks":    total_chunks,
            "model":           OLLAMA_MODEL,
            "embedding_model": OLLAMA_EMBED_MODEL,
        }

    def reinitialize(self, api_key: str):
        """No-op for Ollama — no API key needed."""
        pass
