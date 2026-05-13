# app.py — runs as a small FastAPI service on the Pi
from fastembed import TextEmbedding
from qdrant_edge import EdgeShard, Query, QueryRequest
from fastapi import FastAPI

text_model = TextEmbedding(
    model_name="Qdrant/clip-ViT-B-32-text",
    cache_dir="/mnt/ssd/models",
    local_files_only=True,
)
shard = EdgeShard.create("/mnt/ssd/photo_shard", ...)  # or load existing
app = FastAPI()

@app.get("/search")
def search(q: str, k: int = 20):
    vec = list(text_model.embed([q]))[0].tolist()
    results = shard.query(QueryRequest(
        query=Query.Nearest(vec, using="clip"),
        limit=k,
        with_payload=True,
        with_vector=False,
    ))
    return [{"path": r.payload["path"], "score": r.score} for r in results.points]
