"""
Offline photo search — query app for Raspberry Pi.

A single-file FastAPI service that:
  - Opens a Qdrant EDGE shard built by the indexer.
  - Embeds query text with FastEmbed's CLIP text encoder.
  - Serves a one-page LAN web UI with text search + optional EXIF filters.
  - Streams thumbnails on demand (cached on disk).

Run:
    pip install fastapi uvicorn fastembed qdrant-edge-py pillow
    python photo_search_app.py

Then open http://<pi-ip>:8000 from any device on your LAN.
"""

from __future__ import annotations

import io
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query as Q
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastembed import TextEmbedding
from PIL import Image

from qdrant_edge import (
    EdgeShard,
    Query as EQ,
    QueryRequest,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

# ---------- configuration ----------

SHARD_DIR = os.environ.get("PHOTO_SHARD_DIR", "/mnt/ssd/photo_shard")
MODELS_DIR = os.environ.get("PHOTO_MODELS_DIR", "/mnt/ssd/models")
THUMB_DIR = Path(os.environ.get("PHOTO_THUMB_DIR", "/mnt/ssd/thumbs"))
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
VECTOR_NAME = "clip"
THUMB_SIZE = 384  # long edge in px; balances quality vs Pi network throughput

THUMB_DIR.mkdir(parents=True, exist_ok=True)

# ---------- model + shard, loaded once ----------

print("Loading CLIP text encoder...")
text_model = TextEmbedding(
    model_name=TEXT_MODEL,
    cache_dir=MODELS_DIR,
    local_files_only=True,
)

print(f"Opening EDGE shard at {SHARD_DIR}...")
# The indexer creates the shard; the query app opens the existing one.
# EdgeShard.create() fails if the directory has data, so use the open/load API.
# (The exact call name varies by version; see qdrant-edge docs.)
shard = EdgeShard.open(SHARD_DIR)

# Embedding is CPU-bound and not reentrant in FastEmbed; serialize calls.
_embed_lock = threading.Lock()


def embed_query(text: str) -> list[float]:
    with _embed_lock:
        vec = list(text_model.embed([text]))[0]
    return vec.tolist()


# ---------- FastAPI app ----------

app = FastAPI(title="Photo Search")


@app.get("/api/search")
def search(
    q: str = Q(..., min_length=1, max_length=200),
    k: int = Q(24, ge=1, le=100),
    year: Optional[int] = None,
    after: Optional[str] = None,   # ISO date, e.g. 2023-01-01
    before: Optional[str] = None,
):
    """Text-to-image search with optional EXIF date filters."""
    must: list = []

    if year is not None:
        must.append(FieldCondition(key="year", match=MatchValue(value=year)))

    if after or before:
        rng_kwargs = {}
        if after:
            rng_kwargs["gte"] = datetime.fromisoformat(after).timestamp()
        if before:
            rng_kwargs["lte"] = datetime.fromisoformat(before).timestamp()
        must.append(FieldCondition(key="taken_at", range=Range(**rng_kwargs)))

    flt = Filter(must=must) if must else None
    vec = embed_query(q)

    res = shard.query(
        QueryRequest(
            query=EQ.Nearest(vec, using=VECTOR_NAME),
            filter=flt,
            limit=k,
            with_payload=True,
            with_vector=False,
        )
    )

    return JSONResponse(
        {
            "query": q,
            "count": len(res.points),
            "results": [
                {
                    "id": str(p.id),
                    "score": round(p.score, 4),
                    "path": p.payload.get("path"),
                    "taken_at": p.payload.get("taken_at"),
                    "year": p.payload.get("year"),
                }
                for p in res.points
            ],
        }
    )


@app.get("/api/thumb/{point_id}")
def thumb(point_id: str):
    """Generate (and cache) a thumbnail for one indexed photo."""
    cached = THUMB_DIR / f"{point_id}.jpg"
    if not cached.exists():
        # Look the point up to find the file path; payload-only retrieve.
        res = shard.retrieve(ids=[point_id], with_payload=True, with_vector=False)
        if not res:
            raise HTTPException(404, "unknown id")
        src = Path(res[0].payload["path"])
        if not src.exists():
            raise HTTPException(410, "file gone")
        with Image.open(src) as im:
            im.thumbnail((THUMB_SIZE, THUMB_SIZE))
            im = im.convert("RGB")
            im.save(cached, "JPEG", quality=80, optimize=True)

    return StreamingResponse(open(cached, "rb"), media_type="image/jpeg")


# ---------- one-page UI ----------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Mona+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0B0F19;
    --surface-1: #111824;
    --surface-2: #141A2A;
    --text: #F0F3FA;
    --text-2: #656B7F;
    --border: #4E5366;
    --primary: #DC244C;
    --primary-hover: #CC1845;
    --font: "Mona Sans","Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 16px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  header {
    border-bottom: 1px solid var(--border);
    padding: 24px 32px;
  }
  header h1 {
    margin: 0 0 4px;
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.5px;
  }
  header p {
    margin: 0;
    color: var(--text-2);
    font-size: 14px;
  }
  header .accent { color: var(--primary); }
  main { padding: 32px; max-width: 1110px; margin: 0 auto; }
  .search-row {
    display: flex; gap: 8px; margin-bottom: 16px;
  }
  input[type="text"], input[type="number"], input[type="date"] {
    flex: 1;
    background: var(--surface-1);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: var(--font);
    font-size: 16px;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type="text"]:focus,
  input[type="number"]:focus,
  input[type="date"]:focus {
    border-color: var(--primary);
  }
  button {
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-family: var(--font);
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: var(--primary-hover); }
  .filters {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-bottom: 32px;
  }
  .filters input { flex: 0 0 auto; max-width: 180px; }
  .filters label {
    display: inline-flex; align-items: center; gap: 8px;
    color: var(--text-2); font-size: 14px;
  }
  .meta {
    color: var(--text-2);
    font-size: 14px;
    margin-bottom: 16px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
  }
  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .card:hover { border-color: var(--text-2); }
  .card img {
    display: block;
    width: 100%; aspect-ratio: 1 / 1;
    object-fit: cover;
    background: var(--surface-2);
  }
  .card-meta {
    padding: 8px 12px;
    font-size: 12px;
    color: var(--text-2);
    display: flex; justify-content: space-between;
    border-top: 1px solid var(--border);
  }
  .score {
    color: var(--primary);
    font-variant-numeric: tabular-nums;
  }
  .empty {
    color: var(--text-2);
    text-align: center;
    padding: 64px 16px;
    font-size: 14px;
  }
</style>
</head>
<body>
  <header>
    <h1>Photo Search <span class="accent">·</span> offline</h1>
    <p>Powered by Qdrant EDGE on this Pi. Nothing leaves the house.</p>
  </header>

  <main>
    <form class="search-row" id="form">
      <input type="text" id="q" placeholder='Try: "kayak on the lake at sunset" or "dog wearing a birthday hat"' autofocus>
      <button type="submit">Search</button>
    </form>

    <div class="filters">
      <label>Year <input type="number" id="year" min="1990" max="2099" placeholder="e.g. 2023"></label>
      <label>After <input type="date" id="after"></label>
      <label>Before <input type="date" id="before"></label>
    </div>

    <div class="meta" id="meta"></div>
    <div class="grid" id="grid"></div>
    <div class="empty" id="empty">Type a query above to search your library.</div>
  </main>

<script>
  const form  = document.getElementById('form');
  const qIn   = document.getElementById('q');
  const yIn   = document.getElementById('year');
  const aIn   = document.getElementById('after');
  const bIn   = document.getElementById('before');
  const meta  = document.getElementById('meta');
  const grid  = document.getElementById('grid');
  const empty = document.getElementById('empty');

  async function search(e) {
    if (e) e.preventDefault();
    const q = qIn.value.trim();
    if (!q) return;

    const params = new URLSearchParams({ q, k: 48 });
    if (yIn.value)  params.set('year',   yIn.value);
    if (aIn.value)  params.set('after',  aIn.value);
    if (bIn.value)  params.set('before', bIn.value);

    meta.textContent  = 'Searching…';
    grid.innerHTML    = '';
    empty.style.display = 'none';

    const t0 = performance.now();
    const r  = await fetch('/api/search?' + params.toString());
    const j  = await r.json();
    const ms = Math.round(performance.now() - t0);

    meta.textContent = `${j.count} result${j.count === 1 ? '' : 's'} for "${j.query}" — ${ms} ms`;
    if (j.count === 0) {
      empty.textContent = 'No matches. Try a different phrasing or remove a filter.';
      empty.style.display = 'block';
      return;
    }

    const frag = document.createDocumentFragment();
    for (const p of j.results) {
      const card = document.createElement('div');
      card.className = 'card';
      const taken = p.taken_at
        ? new Date(p.taken_at * 1000).toLocaleDateString()
        : '—';
      card.innerHTML = `
        <a href="/api/thumb/${p.id}" target="_blank">
          <img loading="lazy" src="/api/thumb/${p.id}" alt="">
        </a>
        <div class="card-meta">
          <span>${taken}</span>
          <span class="score">${p.score.toFixed(3)}</span>
        </div>`;
      frag.appendChild(card);
    }
    grid.appendChild(frag);
  }

  form.addEventListener('submit', search);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


if __name__ == "__main__":
    # Bind to all interfaces so other devices on the LAN can reach it.
    uvicorn.run(app, host="0.0.0.0", port=8000)
