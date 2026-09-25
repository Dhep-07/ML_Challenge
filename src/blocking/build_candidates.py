"""
Blocking orchestrator — unions candidate pairs from all strategies.

Architecture (ordered by reliability given sparse pincode/state):
─────────────────────────────────────────────────────────────────
1. Phonetic name code (Metaphone/Soundex) — PRIMARY key
2. Normalized name-prefix                 — SECONDARY exact-ish key
3. City/locality token match              — SUPPLEMENTARY geographic anchor
4. Pincode match                          — OPPORTUNISTIC BONUS (never relied upon)
5. TF-IDF similarity (name + combined)    — MAIN RECALL DRIVER

Country partitioning is soft: partition by exact country match, but
records with blank/unrecognised country fall back to comparison across
all countries.

Output: candidate_pairs.tsv with deduplicated candidates per S1 entity.
"""
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.config import OUTPUT_DIR
from src.blocking.phonetic_blocking import phonetic_blocking
from src.blocking.name_prefix_blocking import name_prefix_blocking
from src.blocking.city_blocking import city_blocking
from src.blocking.pincode_blocking import pincode_blocking
from src.blocking.tfidf_blocking import tfidf_name_blocking, tfidf_combined_blocking

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Country partitioning (soft)
# ---------------------------------------------------------------------------

def _partition_by_country(df_s1, df_other):
    """
    Yield (df_s1_partition, df_other_partition, country_label) tuples.

    Records with blank / NaN country are placed in a special "__ALL__"
    partition that is compared against everything (soft fallback).
    """
    s1_countries = df_s1["country"].fillna("").str.strip().str.upper()
    other_countries = df_other["country"].fillna("").str.strip().str.upper()

    # Identify known countries (non-blank intersection)
    known = set(s1_countries[s1_countries != ""]) | set(other_countries[other_countries != ""])

    for country in known:
        s1_mask = s1_countries == country
        other_mask = other_countries == country
        s1_part = df_s1[s1_mask]
        other_part = df_other[other_mask]
        if len(s1_part) > 0 and len(other_part) > 0:
            yield s1_part, other_part, country

    # Fallback partition: S1 records with blank country vs ALL other records
    s1_blank = df_s1[s1_countries == ""]
    other_blank = df_other[other_countries == ""]
    if len(s1_blank) > 0:
        yield s1_blank, df_other, "__BLANK_S1__"
    # Other records with blank country vs ALL S1 records
    if len(other_blank) > 0:
        yield df_s1, other_blank, "__BLANK_OTHER__"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_candidates(
    df_s1: pd.DataFrame,
    df_other: pd.DataFrame,
    output_path=None,
    tfidf_top_k: int = 20,
    tfidf_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Run all blocking strategies within country partitions, union and
    deduplicate candidate pairs, track per-method provenance, and write
    candidate_pairs.tsv.

    Parameters
    ----------
    df_s1 : DataFrame
        Source-1 records (must have entity_id, business_name_normalized,
        business_address_normalized, country).
    df_other : DataFrame
        Concatenated Source-2 + Source-3 records (same columns).
    output_path : Path or str, optional
        Where to write candidate_pairs.tsv.
    tfidf_top_k : int
        Top-K neighbours per S1 entity for TF-IDF blocking.
    tfidf_threshold : float
        Minimum cosine similarity for TF-IDF candidates.

    Returns
    -------
    DataFrame with columns [source1_entity_id, candidate_entity_ids].
    """
    if output_path is None:
        output_path = Path(OUTPUT_DIR) / "candidate_pairs.tsv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all (s1_id, other_id, method) triples
    all_pairs: List[Tuple[str, str, str]] = []
    # Track per-method counts for diagnostics
    method_counts: Dict[str, int] = defaultdict(int)

    for s1_part, other_part, country_label in _partition_by_country(df_s1, df_other):
        logger.info(
            f"Blocking partition [{country_label}]: "
            f"{len(s1_part)} S1 × {len(other_part)} other"
        )

        # 1. Phonetic name code (PRIMARY)
        pairs = phonetic_blocking(s1_part, other_part)
        all_pairs.extend(pairs)
        method_counts["phonetic"] += len(pairs)

        # 2. Name prefix (SECONDARY)
        pairs = name_prefix_blocking(s1_part, other_part)
        all_pairs.extend(pairs)
        method_counts["name_prefix"] += len(pairs)

        # 3. City token match (SUPPLEMENTARY)
        pairs = city_blocking(s1_part, other_part)
        all_pairs.extend(pairs)
        method_counts["city"] += len(pairs)

        # 4. Pincode match (OPPORTUNISTIC BONUS)
        pairs = pincode_blocking(s1_part, other_part)
        all_pairs.extend(pairs)
        method_counts["pincode"] += len(pairs)

        # 5. TF-IDF name-only (MAIN RECALL DRIVER)
        pairs = tfidf_name_blocking(
            s1_part, other_part, top_k=tfidf_top_k, threshold=tfidf_threshold,
        )
        all_pairs.extend(pairs)
        method_counts["tfidf_name"] += len(pairs)

        # 6. TF-IDF combined name+address (MAIN RECALL DRIVER)
        pairs = tfidf_combined_blocking(
            s1_part, other_part, top_k=tfidf_top_k, threshold=tfidf_threshold,
        )
        all_pairs.extend(pairs)
        method_counts["tfidf_combined"] += len(pairs)

    # ----- Union & deduplicate -----
    # Track which methods caught each pair for diagnostics
    pair_methods: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for s1_id, other_id, method in all_pairs:
        pair_methods[(s1_id, other_id)].add(method)

    # Group candidates per S1 entity
    s1_candidates: Dict[str, Set[str]] = defaultdict(set)
    for (s1_id, other_id) in pair_methods:
        s1_candidates[s1_id].add(other_id)

    # Ensure every S1 entity has a row (even with empty candidates)
    for eid in df_s1["entity_id"]:
        if eid not in s1_candidates:
            s1_candidates[eid] = set()

    # Build output DataFrame
    rows = []
    for s1_id in sorted(s1_candidates):
        cands = sorted(s1_candidates[s1_id])
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands),
        })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, sep="\t", index=False)

    # ----- Diagnostics -----
    total_unique_pairs = len(pair_methods)
    logger.info(f"=== Blocking Summary ===")
    logger.info(f"Total unique candidate pairs: {total_unique_pairs}")
    for method, count in sorted(method_counts.items()):
        logger.info(f"  {method}: {count} raw pairs")
    logger.info(
        f"S1 entities with ≥1 candidate: "
        f"{sum(1 for v in s1_candidates.values() if v)} / {len(s1_candidates)}"
    )
    logger.info(f"Output written to: {output_path}")

    return result_df


# ---------------------------------------------------------------------------
# Per-method recall tracking
# ---------------------------------------------------------------------------

def evaluate_blocking_recall(
    df_s1: pd.DataFrame,
    df_other: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, dict]:
    """
    Measure per-method recall against ground truth.
    Reports recall conditional on field availability (e.g. "recall from
    pincode blocking, among the X% of records that had one" vs
    "overall recall across all records").

    Parameters
    ----------
    df_s1 : preprocessed S1 DataFrame
    df_other : preprocessed S2+S3 concatenated DataFrame
    ground_truth : DataFrame with [source1_entity_id, matched_entity_ids]

    Returns
    -------
    Dict mapping method_name → {recall, pairs_found, true_pairs_recoverable, coverage_pct}
    """
    from src.blocking.pincode_blocking import _extract_pincode
    from src.blocking.city_blocking import _city_key

    # Parse ground truth into set of (s1_id, other_id)
    gt_pairs: Set[Tuple[str, str]] = set()
    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched = str(row.get("matched_entity_ids", "") or "")
        if matched.strip():
            for other_id in matched.split(","):
                other_id = other_id.strip()
                if other_id:
                    gt_pairs.add((s1_id, other_id))

    if not gt_pairs:
        logger.warning("No ground truth pairs found.")
        return {}

    total_gt = len(gt_pairs)

    # Run each strategy and measure recall
    strategies = {
        "phonetic": lambda s1, ot: phonetic_blocking(s1, ot),
        "name_prefix": lambda s1, ot: name_prefix_blocking(s1, ot),
        "city": lambda s1, ot: city_blocking(s1, ot),
        "pincode": lambda s1, ot: pincode_blocking(s1, ot),
        "tfidf_name": lambda s1, ot: tfidf_name_blocking(s1, ot),
        "tfidf_combined": lambda s1, ot: tfidf_combined_blocking(s1, ot),
    }

    results = {}
    for method_name, strategy_fn in strategies.items():
        pairs_raw = []
        for s1_part, other_part, _ in _partition_by_country(df_s1, df_other):
            pairs_raw.extend(strategy_fn(s1_part, other_part))

        found_pairs = {(s1, ot) for s1, ot, _ in pairs_raw}
        recovered = found_pairs & gt_pairs
        recall = len(recovered) / total_gt if total_gt > 0 else 0.0

        # Coverage: what fraction of GT pairs could this method even apply to?
        # For pincode: only pairs where both sides have a pincode
        if method_name == "pincode":
            s1_with_pin = set(df_s1[df_s1["business_address_normalized"].fillna("").apply(
                lambda x: _extract_pincode(x) is not None
            )]["entity_id"])
            other_with_pin = set(df_other[df_other["business_address_normalized"].fillna("").apply(
                lambda x: _extract_pincode(x) is not None
            )]["entity_id"])
            applicable_gt = {(s, o) for s, o in gt_pairs
                             if s in s1_with_pin and o in other_with_pin}
            coverage_pct = len(applicable_gt) / total_gt * 100 if total_gt else 0
            conditional_recall = (
                len(recovered & applicable_gt) / len(applicable_gt)
                if applicable_gt else 0.0
            )
        elif method_name == "city":
            s1_with_city = set(df_s1[df_s1["business_address_normalized"].fillna("").apply(
                lambda x: _city_key(x) != ""
            )]["entity_id"])
            other_with_city = set(df_other[df_other["business_address_normalized"].fillna("").apply(
                lambda x: _city_key(x) != ""
            )]["entity_id"])
            applicable_gt = {(s, o) for s, o in gt_pairs
                             if s in s1_with_city and o in other_with_city}
            coverage_pct = len(applicable_gt) / total_gt * 100 if total_gt else 0
            conditional_recall = (
                len(recovered & applicable_gt) / len(applicable_gt)
                if applicable_gt else 0.0
            )
        else:
            coverage_pct = 100.0
            conditional_recall = recall

        results[method_name] = {
            "recall_overall": round(recall, 4),
            "recall_conditional": round(conditional_recall, 4),
            "pairs_found": len(found_pairs),
            "gt_pairs_recovered": len(recovered),
            "total_gt_pairs": total_gt,
            "coverage_pct": round(coverage_pct, 1),
        }

    # Union recall
    all_found: Set[Tuple[str, str]] = set()
    for method_name, strategy_fn in strategies.items():
        pairs_raw = []
        for s1_part, other_part, _ in _partition_by_country(df_s1, df_other):
            pairs_raw.extend(strategy_fn(s1_part, other_part))
        all_found |= {(s1, ot) for s1, ot, _ in pairs_raw}

    union_recovered = all_found & gt_pairs
    results["UNION_ALL"] = {
        "recall_overall": round(len(union_recovered) / total_gt, 4) if total_gt else 0,
        "recall_conditional": round(len(union_recovered) / total_gt, 4) if total_gt else 0,
        "pairs_found": len(all_found),
        "gt_pairs_recovered": len(union_recovered),
        "total_gt_pairs": total_gt,
        "coverage_pct": 100.0,
    }

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Run build_candidates() with preprocessed DataFrames.")
