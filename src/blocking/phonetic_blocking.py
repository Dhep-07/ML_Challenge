"""
Phonetic name-code blocking (PRIMARY strategy).

Generates Metaphone codes from business_name_normalized and groups records
by identical codes.  Name is present on every record and reasonably robust
to spelling noise, making this the single most reliable exact-key strategy.

Falls back to Soundex when Metaphone returns an empty string (rare but
possible on very short or numeric-only names).
"""
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import jellyfish


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def _tokenize(name: str) -> List[str]:
    """Split on whitespace and strip non-alpha noise tokens."""
    if not name:
        return []
    return [t for t in name.split() if re.search(r'[A-Z]', t)]


def _metaphone_key(name: str) -> str:
    """
    Build a composite Metaphone key from all alphabetic tokens in the name.
    Concatenation of per-token Metaphone codes gives a fingerprint that is
    tolerant to individual-character typos while remaining discriminating
    enough to avoid huge buckets.
    """
    tokens = _tokenize(name)
    if not tokens:
        return ""
    codes = []
    for tok in tokens:
        m = jellyfish.metaphone(tok)
        if m:
            codes.append(m)
    return " ".join(codes)


def _soundex_key(name: str) -> str:
    """Fallback fingerprint when Metaphone gives nothing."""
    tokens = _tokenize(name)
    if not tokens:
        return ""
    codes = []
    for tok in tokens:
        try:
            s = jellyfish.soundex(tok)
            if s:
                codes.append(s)
        except Exception:
            continue
    return " ".join(codes)


def phonetic_key(name: str) -> str:
    """Return a phonetic blocking key for *name* (already uppercased)."""
    key = _metaphone_key(name)
    if not key:
        key = _soundex_key(name)
    return key


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

def phonetic_blocking(
    df_s1,
    df_other,
    name_col: str = "business_name_normalized",
) -> List[Tuple[str, str, str]]:
    """
    Pair every S1 entity with every S2/S3 entity that shares the same
    phonetic key within the same country partition.

    Returns a list of (s1_id, other_id, method_label) tuples.
    """
    method = "phonetic"

    # Build phonetic keys
    s1_keys = df_s1[name_col].fillna("").apply(phonetic_key)
    other_keys = df_other[name_col].fillna("").apply(phonetic_key)

    # Index the "other" side by key for fast lookup
    other_by_key: Dict[str, List[str]] = defaultdict(list)
    for idx, key in other_keys.items():
        if key:
            eid = df_other.at[idx, "entity_id"]
            other_by_key[key].append(eid)

    pairs: List[Tuple[str, str, str]] = []
    for idx, key in s1_keys.items():
        if not key:
            continue
        s1_id = df_s1.at[idx, "entity_id"]
        for other_id in other_by_key.get(key, []):
            pairs.append((s1_id, other_id, method))

    return pairs
