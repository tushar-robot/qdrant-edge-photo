# indexer.py
import uuid, hashlib
from pathlib import Path
from fastembed import ImageEmbedding
from qdrant_edge import EdgeShard, EdgeConfig, Point, UpdateOperation

SHARD_DIR = "/mnt/ssd/photo_shard"
MODELS_DIR = "/mnt/ssd/models"
PHOTOS_DIR = Path("/mnt/nas/photos")
VECTOR_NAME = "clip"

image_model = ImageEmbedding(
    model_name="Qdrant/clip-ViT-B-32-vision",
    cache_dir=MODELS_DIR,
    local_files_only=True,
)

# Open existing shard or create on first run — check the EDGE docs for the
# exact "load existing" call; create() fails if the directory has data.
shard = EdgeShard.create(SHARD_DIR, EdgeConfig(...))  # configure 512-dim + scalar quant

def path_id(p: Path) -> str:
    return str(uuid.UUID(hashlib.md5(str(p).encode()).hexdigest()))

# Batch for throughput
batch_paths = []
BATCH = 16

for p in PHOTOS_DIR.rglob("*.jpg"):
    batch_paths.append(p)
    if len(batch_paths) == BATCH:
        vecs = list(image_model.embed(batch_paths))
        points = [
            Point(
                id=path_id(p),
                vector={VECTOR_NAME: v.tolist()},
                payload={"path": str(p), "mtime": p.stat().st_mtime},
            )
            for p, v in zip(batch_paths, vecs)
        ]
        shard.update(UpdateOperation.upsert_points(points))
        batch_paths.clear()

shard.optimize()  # EDGE has no background optimizer — call it yourself
shard.close()
