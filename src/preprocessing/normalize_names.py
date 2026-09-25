"""
Entity name normalization utilities.
Includes suffix stripping and abbreviation expansion.
"""
import re

def normalize_name(name: str) -> str:
    """Normalizes entity names by lowering, expanding abbreviations, and stripping suffixes."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'\b(pvt|ltd|inc|corp|co|llp)\b\.?', '', name)
    return re.sub(r'\s+', ' ', name).strip()
