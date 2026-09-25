"""
Entity name normalization calling the 3-step preprocessing pipeline.
"""
from src.preprocessing.pipeline import preprocess_text

def normalize_name(name: str) -> str:
    """Normalizes entity names through the 3-step preprocessing pipeline."""
    return preprocess_text(name)
