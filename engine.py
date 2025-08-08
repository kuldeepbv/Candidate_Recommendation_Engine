
import os
import re
import io
import json
import faiss
import time
import pickle
import string
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

# Lightweight PDF/DOCX/TXT readers
from PyPDF2 import PdfReader
import docx2txt

# NLTK for cleaning/lemmatization (auto-downloads on first run)
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Embeddings
from sentence_transformers import SentenceTransformer

# Optional Gemini summary
GEMINI_ENABLED = False
try:
    import google.generativeai as genai
    
    # Directly hard-code your API key here
    GEMINI_API_KEY = "AIzaSyAB9UyYH9c8oGAYdWESqaxHNt2m5hqeT3k"  # <-- replace with your actual key

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
except Exception:
    GEMINI_ENABLED = False


# -----------------------------
# Storage paths (local, persisted)
# -----------------------------
APP_DIR = Path(os.environ.get("FAISS_APP_DIR", ".")).resolve()
DATA_DIR = APP_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
META_PATH = DATA_DIR / "metadata.pkl"
INDEX_PATH = INDEX_DIR / "index.faiss"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Embedding model
# -----------------------------
# Default: small/fast model. You can swap to 'all-mpnet-base-v2' if you want higher quality
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
_model_cache = None

def get_embedder():
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_cache

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embs


# -----------------------------
# Basic text extraction
# -----------------------------
def _read_pdf_bytes(b: bytes) -> str:
    reader = PdfReader(io.BytesIO(b))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()

def _read_pdf(path: Union[str, Path]) -> str:
    with open(path, "rb") as f:
        return _read_pdf_bytes(f.read())

def _read_docx_bytes(b: bytes) -> str:
    tmp_path = DATA_DIR / f"_tmp_{time.time_ns()}.docx"
    with open(tmp_path, "wb") as f:
        f.write(b)
    try:
        txt = docx2txt.process(str(tmp_path)) or ""
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    return txt.strip()

def _read_docx(path: Union[str, Path]) -> str:
    return docx2txt.process(str(path)) or ""

def _read_txt_bytes(b: bytes) -> str:
    # Attempt utf-8 then latin-1
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")

def _read_txt(path: Union[str, Path]) -> str:
    with open(path, "rb") as f:
        return _read_txt_bytes(f.read())

def extract_text_from_file(file_or_path: Union[str, Path, bytes, io.BytesIO], filename: Optional[str] = None) -> str:
    """
    Extract text from PDF/DOCX/TXT.
    - Accepts a filesystem path (str/Path) OR raw bytes/BytesIO (e.g., from Streamlit uploader).
    - If bytes are passed, provide `filename` for extension detection.
    """
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            return _read_pdf(path)
        elif ext in [".docx", ".doc"]:
            return _read_docx(path)
        elif ext in [".txt", ""]:
            return _read_txt(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    else:
        # bytes-like
        if isinstance(file_or_path, io.BytesIO):
            b = file_or_path.getvalue()
        elif isinstance(file_or_path, (bytes, bytearray)):
            b = bytes(file_or_path)
        else:
            raise ValueError("file_or_path must be path, bytes, or BytesIO")

        if not filename:
            raise ValueError("When passing bytes, you must provide `filename` to detect file type.")

        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return _read_pdf_bytes(b)
        elif ext in [".docx", ".doc"]:
            return _read_docx_bytes(b)
        elif ext in [".txt", ""]:
            return _read_txt_bytes(b)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# Cleaning & Lemmatization
# -----------------------------
_lemmatizer = None
_stopwords = None

def _ensure_nltk():
    global _lemmatizer, _stopwords
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    if _stopwords is None:
        _stopwords = set(stopwords.words("english"))

def clean__and_lemmatize(text: str) -> str:
    """
    Basic cleaning: lowercase, remove URLs, non-alphanumerics, stopwords; then lemmatize.
    """
    _ensure_nltk()
    text = text.lower()
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in _stopwords and len(t) > 2]
    lemmas = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


# -----------------------------
# Chunking
# -----------------------------
def chunk_text_by_words(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks


# -----------------------------
# FAISS Store (cosine via inner product on normalized vectors)
# -----------------------------
class FAISSStore:
    def __init__(self, dim: int, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index = faiss.IndexFlatIP(dim)  # inner product ~= cosine if vectors are normalized
        self.metadata: Dict[str, Dict] = {}  # resume_id -> {path, filename, cleaned_text, original_text, embedding (np.ndarray)}

    def add(self, resume_id: str, embedding: np.ndarray, meta: Dict):
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {embedding.shape[1]}")
        self.index.add(embedding.astype("float32"))
        # Store embedding as float32 for compactness
        meta = dict(meta)
        meta["embedding"] = embedding.astype("float32")[0]
        self.metadata[resume_id] = meta

    def ids(self) -> List[str]:
        return list(self.metadata.keys())

    def get_meta(self, resume_id: str) -> Dict:
        return self.metadata.get(resume_id, {})

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump({"dim": self.dim, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        if not index_path.exists() or not meta_path.exists():
            return None
        with open(meta_path, "rb") as f:
            obj = pickle.load(f)
        dim = obj.get("dim")
        store = cls(dim, index_path=index_path, meta_path=meta_path)
        store.index = faiss.read_index(str(index_path))
        store.metadata = obj.get("metadata", {})
        return store


# -----------------------------
# Business functions
# -----------------------------
def store_resume_embedding(resume_id: str, path_or_bytes: Union[str, Path, bytes, io.BytesIO], filename: Optional[str] = None) -> Dict:
    """
    1) Extract original_text
    2) Clean & lemmatize
    3) Chunk and embed chunks
    4) Mean-pool chunk embeddings -> single vector
    5) Save to FAISS + metadata
    Returns the stored metadata for the resume
    """
    original_text = extract_text_from_file(path_or_bytes, filename=filename)
    cleaned = clean__and_lemmatize(original_text)
    chunks = chunk_text_by_words(cleaned, chunk_size=200, overlap=30)
    if not chunks:
        raise ValueError("No text extracted from resume.")

    chunk_embs = embed_texts(chunks)  # normalized
    mean_emb = np.mean(chunk_embs, axis=0)
    # normalize mean vector to unit length
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    # Load or init store
    probe = embed_texts(["probe"])[0]
    dim = probe.shape[0]
    store = FAISSStore.load()
    if store is None:
        store = FAISSStore(dim)

    meta = {
        "resume_id": resume_id,
        "path": str(path_or_bytes) if isinstance(path_or_bytes, (str, Path)) else None,
        "filename": filename if filename else (str(path_or_bytes) if isinstance(path_or_bytes, (str, Path)) else None),
        "cleaned_text": cleaned,
        "original_text": original_text,
    }
    store.add(resume_id, mean_emb, meta)
    store.save()
    return store.get_meta(resume_id)


def compute_resume_score_against_jd(jd_chunks: List[str], resume_embedding: np.ndarray) -> float:
    """
    Embed each JD chunk; compute cosine similarity with the (already normalized) resume embedding,
    then average the per-chunk similarities.
    """
    if resume_embedding.ndim == 1:
        resume_embedding = resume_embedding.reshape(1, -1)
    jd_embs = embed_texts(jd_chunks)  # normalized
    # Cosine via inner product of normalized vectors
    sims = np.dot(jd_embs, resume_embedding.T).flatten()
    return float(np.mean(sims))


def rank_resumes(jd_text: str, top_n: int = 5) -> pd.DataFrame:
    """
    Clean JD -> chunk -> score vs every stored resume -> return df(resume_id, score) sorted desc.
    """
    store = FAISSStore.load()
    if store is None or len(store.ids()) == 0:
        raise RuntimeError("No resumes indexed yet. Please add resumes first.")

    cleaned_jd = clean__and_lemmatize(jd_text)
    jd_chunks = chunk_text_by_words(cleaned_jd, chunk_size=200, overlap=30)
    if not jd_chunks:
        raise ValueError("Job description has no usable text after cleaning.")

    rows = []
    for rid in store.ids():
        meta = store.get_meta(rid)
        emb = meta.get("embedding")
        if emb is None:
            continue
        score = compute_resume_score_against_jd(jd_chunks, np.array(emb))
        rows.append({"resume_id": rid, "score": score, "filename": meta.get("filename")})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def _gemini_summary(jd_text: str, resume_text: str) -> str:
    prompt = f"""
You are an assistant that writes a concise, specific, bullet-style fit summary.
Compare the job description and the candidate resume.
Highlight 4–6 concrete reasons the candidate is a good fit. Mention relevant skills, projects, and outcomes.
Keep it under 120 words.

JOB DESCRIPTION:
{jd_text[:4000]}

RESUME:
{resume_text[:4000]}
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    resp = model.generate_content(prompt)
    return resp.text.strip()


def _fallback_summary(jd_text: str, resume_text: str) -> str:
    # Super-simple fallback: pick top overlapping keywords as bullets
    jd_words = set(clean__and_lemmatize(jd_text).split())
    res_words = set(clean__and_lemmatize(resume_text).split())
    overlap = list(jd_words.intersection(res_words))[:10]
    bullets = "\n".join([f"- Matches keyword: {w}" for w in overlap]) or "- Relevant experience aligned with JD keywords."
    return f"Quick fit summary (fallback):\n{bullets}"


def generate_fit_summary_gemini(jd_text: str, resume_text: str) -> str:
    """
    Uses Gemini if GEMINI_API_KEY is present, else a deterministic fallback summary.
    """
    if GEMINI_ENABLED:
        try:
            return _gemini_summary(jd_text, resume_text)
        except Exception as e:
            return _fallback_summary(jd_text, resume_text) + f"\n(Note: Gemini error: {e})"
    else:
        return _fallback_summary(jd_text, resume_text)


def final_output_with_summaries(rank_df: pd.DataFrame, jd_text: str) -> pd.DataFrame:
    """
    For each resume_id in rank_df, fetch stored resume text and generate a fit summary.
    Returns a DataFrame with columns: resume_id, filename, score, summary
    """
    store = FAISSStore.load()
    if store is None:
        raise RuntimeError("No resumes indexed yet.")
    out_rows = []
    for _, row in rank_df.iterrows():
        rid = row["resume_id"]
        meta = store.get_meta(rid)
        resume_text = meta.get("original_text") or meta.get("cleaned_text") or ""
        summary = generate_fit_summary_gemini(jd_text, resume_text)
        out_rows.append({
            "resume_id": rid,
            "filename": meta.get("filename"),
            "score": round(float(row["score"]), 4),
            "summary": summary,
        })
    return pd.DataFrame(out_rows)


# Convenience: clear index (for testing)
def clear_index():
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if META_PATH.exists():
        META_PATH.unlink()
    # re-create empty dirs
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return True
