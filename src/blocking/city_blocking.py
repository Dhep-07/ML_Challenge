"""
City / locality token-match blocking (SUPPLEMENTARY strategy).

Extracts city-like tokens from the address and uses them as a
geographic anchor.  Skips gracefully when absent — contributes
nothing rather than excluding the record.

Heuristic city extraction: take the last substantive comma-segment
or the last 1-2 capitalised tokens that are not common address noise
(road types, unit numbers, etc.).
"""
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


# Common address noise tokens to filter out during city extraction
_ADDRESS_NOISE = {
    "ROAD", "RD", "STREET", "ST", "AVENUE", "AVE", "BOULEVARD", "BLVD",
    "LANE", "LN", "DRIVE", "DR", "COURT", "CT", "PLACE", "PL",
    "SUITE", "STE", "FLOOR", "FL", "UNIT", "APT", "APARTMENT",
    "BUILDING", "BLDG", "BLOCK", "PLOT", "SECTOR", "SEC",
    "NEAR", "OPPOSITE", "OPP", "BEHIND", "BESIDE", "NEXT",
    "NORTH", "SOUTH", "EAST", "WEST", "NE", "NW", "SE", "SW",
    "NO", "NUMBER", "NULL", "NONE", "NA", "ATM", "SBI",
    "MAIN", "CROSS", "CIRCLE", "SQUARE", "PARK", "MARKET",
    "PHASE", "INDUSTRIAL", "AREA", "ZONE", "ESTATE", "NAGAR",
    "COLONY", "GARDEN", "GARDENS", "HEIGHTS", "TOWER", "TOWERS",
}


def _extract_city_tokens(address: str) -> List[str]:
    """
    Best-effort extraction of city/locality tokens from a business address.

    Strategy:
    1. Split by comma → take the last segment that has alphabetic content
       (often city, state or city alone).
    2. From that segment, keep only alpha tokens ≥ 3 chars that are not
       address noise words.
    3. Return up to 2 tokens (to keep buckets reasonably sized).

    Returns an empty list when nothing useful can be extracted — this is
    expected and fine; the record simply gets no city block.
    """
    if not address:
        return []

    segments = [s.strip() for s in address.split(",") if s.strip()]
    if not segments:
        return []

    # Walk segments from the end looking for one with alpha content
    for seg in reversed(segments):
        tokens = re.findall(r'[A-Z]{3,}', seg.upper())
        useful = [t for t in tokens if t not in _ADDRESS_NOISE]
        if useful:
            return useful[:2]  # cap at 2 to avoid overly specific keys

    return []


def _city_key(address: str) -> str:
    """Return a city blocking key or empty string."""
    tokens = _extract_city_tokens(address)
    return " ".join(sorted(tokens)) if tokens else ""


def city_blocking(
    df_s1,
    df_other,
    address_col: str = "business_address_normalized",
) -> List[Tuple[str, str, str]]:
    """
    Pair S1 entities with S2/S3 entities sharing city/locality tokens.
    Records without extractable city tokens are silently skipped.

    Returns a list of (s1_id, other_id, method_label) tuples.
    """
    method = "city"

    s1_keys = df_s1[address_col].fillna("").apply(_city_key)
    other_keys = df_other[address_col].fillna("").apply(_city_key)

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
