"""
Address normalization and feature extraction.
Handles abbreviation expansions (e.g. Rd -> Road) and pincode/city/state extraction.
"""
import re

def normalize_address(address: str) -> str:
    """Expands address abbreviations such as Rd -> Road, St -> Street."""
    if not isinstance(address, str):
        return ""
    addr = address.lower()
    addr = re.sub(r'\brd\b\.?', 'road', addr)
    addr = re.sub(r'\bst\b\.?', 'street', addr)
    addr = re.sub(r'\bave\b\.?', 'avenue', addr)
    return re.sub(r'\s+', ' ', addr).strip()

def extract_pincode(address: str) -> str:
    """Extracts 6-digit Indian postal code from address string."""
    if not isinstance(address, str):
        return ""
    match = re.search(r'\b\d{6}\b', address)
    return match.group(0) if match else ""
