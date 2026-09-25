"""
Exact key blocking strategies:
- Pincode matching
- Phonetic hashing
- Normalized name key blocking
"""
def generate_exact_blocks(df):
    """Generates block keys based on exact attributes (e.g. pincode, phonetic keys)."""
    blocks = {}
    for idx, row in df.iterrows():
        key = row.get("pincode", "") or row.get("name_key", "")
        if key:
            blocks.setdefault(key, []).append(idx)
    return blocks
