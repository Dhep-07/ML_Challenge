"""
Smoke test for the blocking module using synthetic data.
Verifies that all strategies produce expected pairs and the
union/deduplication works correctly.
"""
import pandas as pd
from src.blocking import (
    build_candidates,
    phonetic_blocking,
    name_prefix_blocking,
    city_blocking,
    pincode_blocking,
    tfidf_name_blocking,
    tfidf_combined_blocking,
    phonetic_key,
)


def _make_test_data():
    """Create minimal synthetic datasets for testing."""
    s1 = pd.DataFrame([
        {"entity_id": "S1-001", "business_name": "Acme Pvt Ltd",
         "business_name_normalized": "ACME PRIVATE LIMITED",
         "business_address": "12 High Road, Mumbai 400001",
         "business_address_normalized": "12 HIGH ROAD, MUMBAI 400001",
         "country": "India"},
        {"entity_id": "S1-002", "business_name": "Global Tech Corp",
         "business_name_normalized": "GLOBAL TECHNOLOGY CORPORATION",
         "business_address": "456 Park Avenue, New York, NY 10001",
         "business_address_normalized": "456 PARK AVENUE, NEW YORK, NY 10001",
         "country": "US"},
        {"entity_id": "S1-003", "business_name": "Sunrise Industries",
         "business_name_normalized": "SUNRISE INDUSTRIES",
         "business_address": "Near SBI ATM",
         "business_address_normalized": "NEAR SBI ATM",
         "country": "India"},
    ])

    other = pd.DataFrame([
        {"entity_id": "S2-010", "business_name": "Acme Private Limited",
         "business_name_normalized": "ACME PRIVATE LIMITED",
         "business_address": "12 High Rd, Mumbai 400001",
         "business_address_normalized": "12 HIGH ROAD, MUMBAI 400001",
         "country": "India"},
        {"entity_id": "S3-020", "business_name": "Global Tech Corporation",
         "business_name_normalized": "GLOBAL TECHNOLOGY CORPORATION",
         "business_address": "456 Park Ave, NY 10001",
         "business_address_normalized": "456 PARK AVENUE, NY 10001",
         "country": "US"},
        {"entity_id": "S2-030", "business_name": "Totally Different LLC",
         "business_name_normalized": "TOTALLY DIFFERENT LLC",
         "business_address": "789 Other Street, Chicago, IL 60601",
         "business_address_normalized": "789 OTHER STREET, CHICAGO, IL 60601",
         "country": "US"},
        {"entity_id": "S3-040", "business_name": "Sunrise Indstries",  # typo
         "business_name_normalized": "SUNRISE INDSTRIES",
         "business_address": "",
         "business_address_normalized": "",
         "country": "India"},
    ])

    return s1, other


def test_phonetic_blocking():
    s1, other = _make_test_data()
    pairs = phonetic_blocking(s1, other)
    pair_set = {(s, o) for s, o, _ in pairs}
    # Acme -> Acme should match phonetically
    assert ("S1-001", "S2-010") in pair_set, f"Expected S1-001/S2-010 in phonetic pairs, got {pair_set}"
    print("  [PASS] phonetic_blocking")


def test_name_prefix_blocking():
    s1, other = _make_test_data()
    pairs = name_prefix_blocking(s1, other)
    pair_set = {(s, o) for s, o, _ in pairs}
    # ACME prefix should match
    assert ("S1-001", "S2-010") in pair_set, f"Expected S1-001/S2-010 in prefix pairs, got {pair_set}"
    print("  [PASS] name_prefix_blocking")


def test_pincode_blocking():
    s1, other = _make_test_data()
    pairs = pincode_blocking(s1, other)
    pair_set = {(s, o) for s, o, _ in pairs}
    # 400001 in Mumbai should match
    assert ("S1-001", "S2-010") in pair_set, f"Expected S1-001/S2-010 in pincode pairs, got {pair_set}"
    # 10001 should match
    assert ("S1-002", "S3-020") in pair_set, f"Expected S1-002/S3-020 in pincode pairs"
    # S1-003 has no pincode — should NOT appear
    s1_003_pairs = [p for p in pairs if p[0] == "S1-003"]
    assert len(s1_003_pairs) == 0, f"S1-003 has no pincode, should have no pairs, got {s1_003_pairs}"
    print("  [PASS] pincode_blocking (including graceful skip)")


def test_city_blocking():
    s1, other = _make_test_data()
    pairs = city_blocking(s1, other)
    pair_set = {(s, o) for s, o, _ in pairs}
    # Mumbai should match
    assert ("S1-001", "S2-010") in pair_set, f"Expected S1-001/S2-010 in city pairs, got {pair_set}"
    print("  [PASS] city_blocking")


def test_tfidf_name_blocking():
    s1, other = _make_test_data()
    pairs = tfidf_name_blocking(s1, other, top_k=5, threshold=0.1)
    pair_set = {(s, o) for s, o, _ in pairs}
    # Near-identical names should definitely appear
    assert ("S1-001", "S2-010") in pair_set, f"Expected S1-001/S2-010 in tfidf_name pairs"
    assert ("S1-002", "S3-020") in pair_set, f"Expected S1-002/S3-020 in tfidf_name pairs"
    print("  [PASS] tfidf_name_blocking")


def test_tfidf_combined_blocking():
    s1, other = _make_test_data()
    pairs = tfidf_combined_blocking(s1, other, top_k=5, threshold=0.1)
    pair_set = {(s, o) for s, o, _ in pairs}
    assert ("S1-001", "S2-010") in pair_set, f"Expected S1-001/S2-010 in tfidf_combined pairs"
    print("  [PASS] tfidf_combined_blocking")


def test_build_candidates_integration():
    s1, other = _make_test_data()
    import tempfile, os
    out = os.path.join(tempfile.gettempdir(), "test_candidate_pairs.tsv")
    result = build_candidates(s1, other, output_path=out)

    # Every S1 entity must have a row
    assert set(result["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003"}

    # S1-001 must have S2-010 as a candidate (caught by multiple methods)
    row_001 = result[result["source1_entity_id"] == "S1-001"].iloc[0]
    assert "S2-010" in row_001["candidate_entity_ids"], \
        f"S2-010 missing from S1-001 candidates: {row_001['candidate_entity_ids']}"

    # S1-002 must have S3-020
    row_002 = result[result["source1_entity_id"] == "S1-002"].iloc[0]
    assert "S3-020" in row_002["candidate_entity_ids"], \
        f"S3-020 missing from S1-002 candidates: {row_002['candidate_entity_ids']}"

    # Output file must be valid TSV
    loaded = pd.read_csv(out, sep="\t")
    assert list(loaded.columns) == ["source1_entity_id", "candidate_entity_ids"]

    os.remove(out)
    print("  [PASS] build_candidates integration")


def test_phonetic_key_examples():
    # Verify phonetic key generation
    k1 = phonetic_key("ACME PRIVATE LIMITED")
    k2 = phonetic_key("ACME PRIVATE LIMITED")
    assert k1 == k2, "Identical names should produce identical keys"
    assert k1 != "", "Key should not be empty"
    print(f"  [PASS] phonetic_key (key={k1!r})")


if __name__ == "__main__":
    print("Running blocking module smoke tests...\n")
    test_phonetic_key_examples()
    test_phonetic_blocking()
    test_name_prefix_blocking()
    test_pincode_blocking()
    test_city_blocking()
    test_tfidf_name_blocking()
    test_tfidf_combined_blocking()
    test_build_candidates_integration()
    print("\nAll blocking smoke tests passed!")
