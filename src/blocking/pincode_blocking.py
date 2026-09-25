"""
Pincode-match blocking (OPPORTUNISTIC BONUS strategy).

When both sides happen to have a pincode/zip, it is a very strong signal.
But most records will NOT have one, and the pipeline must never depend on it.
Records without a pincode simply contribute no pairs from this method.

Pincode extraction heuristic:
- US: 5-digit or 5+4 ZIP  (e.g. 10001, 10001-1234)
- India: 6-digit PIN      (e.g. 400001)
- France: 5-digit code     (e.g. 75001)
- Generic: any standalone 5-6 digit number in the address
"""
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


def _extract_pincode(address: str) -> Optional[str]:
    """
    Extract the first plausible postal code from *address*.
    Returns the raw digit string (no hyphens), or None.
    """
    if not address:
        return None

    # Match 5-6 digit sequences that stand alone (word boundaries)
    # Also catches US ZIP+4 format (strip the extension)
    matches = re.findall(r'\b(\d{5,6})(?:-\d{4})?\b', address)
    for m in matches:
        # Filter out obviously non-postal numbers (years, house numbers < 5 digits standalone)
        if len(m) >= 5:
            return m
    return None


def pincode_blocking(
    df_s1,
    df_other,
    address_col: str = "business_address_normalized",
) -> List[Tuple[str, str, str]]:
    """
    Pair S1 entities with S2/S3 entities sharing the same pincode.
    Records without an extractable pincode are silently skipped.

    Returns a list of (s1_id, other_id, method_label) tuples.
    """
    method = "pincode"

    s1_pins = df_s1[address_col].fillna("").apply(_extract_pincode)
    other_pins = df_other[address_col].fillna("").apply(_extract_pincode)

    other_by_pin: Dict[str, List[str]] = defaultdict(list)
    for idx, pin in other_pins.items():
        if pin:
            eid = df_other.at[idx, "entity_id"]
            other_by_pin[pin].append(eid)

    pairs: List[Tuple[str, str, str]] = []
    for idx, pin in s1_pins.items():
        if not pin:
            continue
        s1_id = df_s1.at[idx, "entity_id"]
        for other_id in other_by_pin.get(pin, []):
            pairs.append((s1_id, other_id, method))

    return pairs
