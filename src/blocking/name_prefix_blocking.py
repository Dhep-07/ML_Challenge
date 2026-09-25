"""
Normalized name-prefix blocking (SECONDARY strategy).

Cheap, fast exact-key strategy: take the first N characters of the
uppercased, whitespace-collapsed business name as a blocking key.
This catches cases where Metaphone diverges due to leading word
differences but the raw prefix still matches.
"""
import re
from collections import defaultdict
from typing import List, Tuple, Dict


def _name_prefix_key(name: str, prefix_len: int = 5) -> str:
    """
    Return the first *prefix_len* alphabetic characters of *name*,
    uppercased and stripped of non-alpha chars.
    """
    if not name:
        return ""
    alpha_only = re.sub(r'[^A-Z]', '', name.upper())
    return alpha_only[:prefix_len] if len(alpha_only) >= prefix_len else alpha_only


def name_prefix_blocking(
    df_s1,
    df_other,
    name_col: str = "business_name_normalized",
    prefix_len: int = 5,
) -> List[Tuple[str, str, str]]:
    """
    Pair every S1 entity with every S2/S3 entity sharing the same
    name prefix key.

    Returns a list of (s1_id, other_id, method_label) tuples.
    """
    method = "name_prefix"

    s1_keys = df_s1[name_col].fillna("").apply(lambda x: _name_prefix_key(x, prefix_len))
    other_keys = df_other[name_col].fillna("").apply(lambda x: _name_prefix_key(x, prefix_len))

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
