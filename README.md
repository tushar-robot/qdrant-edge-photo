# Qdrant Edge Photo Search

Offline semantic photo search for a Raspberry Pi. Search your photo library with natural language queries like "dog wearing a birthday hat" or "kayak on the lake at sunset" — no cloud, no internet, no data leaving your network.

Uses CLIP embeddings and [Qdrant EDGE](https://qdrant.tech/blog/qdrant-edge/) (an embedded, single-shard vector database) for fast on-device similarity search.

![search UI screenshot](docs/screenshot.png)

## How it works

1. **Index** — run `indexer.py` on any machine to compute CLIP image embeddings for your photo library and store them in a Qdrant EDGE shard
2. **Serve** — copy the shard and models to your Pi, then run `photo_search_app.py`
3. **Search** — open `http://<pi-ip>:8000` from any device on your LAN

## Requirements

- Python 3.8+
- Internet access on the indexing machine (one-time model download)
- Raspberry Pi (or any low-power Linux box) for serving

## Setup

### 1. Download models

Run this once on a machine with internet access:

```bash
pip install fastembed
python download_models.py
```

This saves the CLIP models to `./models`. Copy the `models/` directory to your Pi at `/mnt/ssd/models`.

### 2. Install dependencies

```bash
pip install fastapi uvicorn fastembed qdrant-edge-py pillow
```

### 3. Index your photos

```bash
python indexer.py
```

By default this reads photos from `/mnt/nas/photos` and writes the Qdrant EDGE shard to `/mnt/ssd/photo_shard`. Copy the shard to your Pi if indexing on a separate machine.

### 4. Run the search app

```bash
python photo_search_app.py
```

The app starts on `0.0.0.0:8000`. Open `http://<pi-ip>:8000` from any browser on your network.

## Configuration

All paths are set via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PHOTO_SHARD_DIR` | `/mnt/ssd/photo_shard` | Qdrant EDGE shard directory |
| `PHOTO_MODELS_DIR` | `/mnt/ssd/models` | CLIP model cache |
| `PHOTO_THUMB_DIR` | `/mnt/ssd/thumbs` | Generated thumbnail cache |

Example:

```bash
PHOTO_SHARD_DIR=./shard PHOTO_MODELS_DIR=./models PHOTO_THUMB_DIR=./thumbs python photo_search_app.py
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `GET /api/search?q=<query>&k=<n>&year=<year>&after=<date>&before=<date>` | Search (date format: ISO 8601) |
| `GET /api/thumb/{point_id}` | Thumbnail (generated on demand, cached to disk) |

## Storage layout

```
/mnt/ssd/
├── models/          # CLIP model weights (copy once from indexing machine)
├── photo_shard/     # Qdrant EDGE vector index
└── thumbs/          # Thumbnail cache (auto-generated)

/mnt/nas/photos/     # Original photo library (NAS mount or local)
```

## Tech stack

- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- [FastEmbed](https://github.com/qdrant/fastembed) — local CLIP inference
- [Qdrant EDGE](https://qdrant.tech/blog/qdrant-edge/) — embedded vector database
- [Pillow](https://python-pillow.org/) — thumbnail generation
- Vanilla JS/HTML frontend (no build step)
