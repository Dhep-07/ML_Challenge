"""
DEPRECATED — this module has been replaced by the individual strategy modules:
- phonetic_blocking.py  (Metaphone/Soundex)
- name_prefix_blocking.py
- city_blocking.py
- pincode_blocking.py

Kept for backwards compatibility only.
"""
# Re-export from new modules for any legacy imports
from src.blocking.phonetic_blocking import phonetic_blocking
from src.blocking.pincode_blocking import pincode_blocking


def generate_exact_blocks(df):
    """Legacy stub — use the individual strategy modules instead."""
    import warnings
    warnings.warn(
        "generate_exact_blocks() is deprecated. "
        "Use phonetic_blocking(), name_prefix_blocking(), etc. instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {}
