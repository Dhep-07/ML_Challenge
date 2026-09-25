"""
Address normalization and feature extraction calling the 3-step preprocessing pipeline.
"""
import re
from src.preprocessing.pipeline import preprocess_text

def normalize_address(address: str) -> str:
    """Normalizes addresses through the 3-step preprocessing pipeline."""
    return preprocess_text(address)

def extract_pincode(address: str) -> str:
    """Extracts 6-digit Indian postal code from address string."""
    if not isinstance(address, str):
        return ""
    match = re.search(r'\b\d{6}\b', address)
    return match.group(0) if match else ""
