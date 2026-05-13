# download_models.py — run on a machine with internet, then copy ./models to the Pi
from fastembed import ImageEmbedding, TextEmbedding

MODELS_DIR = "./models"
ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision", cache_dir=MODELS_DIR)
TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text", cache_dir=MODELS_DIR)
