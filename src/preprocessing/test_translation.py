# -*- coding: utf-8 -*-
"""
Test script for the mixed-language translation module.
Tests segment splitting, script detection, and the full translation pipeline.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.preprocessing.translate import (
    _split_mixed_text,
    _detect_script,
    _looks_non_english,
    translate_to_english,
)


def test_split_mixed_text():
    print("=== Test: Mixed-text segment splitting ===")

    # Pure ASCII — single Latin segment
    segments = _split_mixed_text("Acme Inc 123 Main")
    assert len(segments) == 1
    assert segments[0][1] == False  # is_non_latin = False
    print("  [PASS] Pure ASCII → single Latin segment")

    # Pure non-Latin
    segments = _split_mixed_text("के पास")
    assert len(segments) == 1
    assert segments[0][1] == True  # is_non_latin = True
    print("  [PASS] Pure Devanagari → single non-Latin segment")

    # Mixed: English + Hindi + English
    segments = _split_mixed_text("ABC Company के पास Main Road")
    assert len(segments) >= 3  # at least 3 segments
    latin_parts = [s for s, nl in segments if not nl]
    non_latin_parts = [s for s, nl in segments if nl]
    assert len(latin_parts) >= 1
    assert len(non_latin_parts) >= 1
    print(f"  [PASS] Mixed Hindi/English → {len(segments)} segments")
    for seg, is_nl in segments:
        label = "NON-LATIN" if is_nl else "LATIN"
        print(f"         [{label}] {seg!r}")


def test_script_detection():
    print("\n=== Test: Script detection ===")

    assert _detect_script("Hello World") is None
    print("  [PASS] ASCII → None")

    assert _detect_script("के पास") == "DEVANAGARI"
    print("  [PASS] Devanagari detected")

    assert _detect_script("Société") == "LATIN_EXTENDED"
    print("  [PASS] Accented Latin detected")

    assert _detect_script("Москва") == "CYRILLIC"
    print("  [PASS] Cyrillic detected")

    assert _detect_script("東京") == "CJK"
    print("  [PASS] CJK detected")


def test_looks_non_english():
    print("\n=== Test: Non-English detection ===")

    assert _looks_non_english("Hello") == False
    assert _looks_non_english("ABC 123") == False
    assert _looks_non_english("के पास") == True
    assert _looks_non_english("Société") == True
    print("  [PASS] All checks correct")


def test_translate_pure_ascii():
    print("\n=== Test: Pure ASCII passthrough ===")
    result = translate_to_english("Acme Inc 123 Main")
    assert result == "Acme Inc 123 Main"
    print(f"  [PASS] '{result}'")


def test_translate_empty():
    print("\n=== Test: Empty/None handling ===")
    assert translate_to_english("") == ""
    assert translate_to_english(None) == ""
    assert translate_to_english("   ") == ""
    print("  [PASS] All empty cases return ''")


def test_translate_mixed_hindi():
    print("\n=== Test: Mixed Hindi+English translation ===")
    text = "ABC Company के पास Main Road"
    result = translate_to_english(text)
    print(f"  Input:  {text}")
    print(f"  Output: {result}")
    # At minimum, the English parts should be preserved
    assert "ABC" in result or "abc" in result.lower()
    assert "Main" in result or "main" in result.lower() or "MAIN" in result.upper()
    print("  [PASS] English parts preserved in output")


if __name__ == "__main__":
    test_split_mixed_text()
    test_script_detection()
    test_looks_non_english()
    test_translate_pure_ascii()
    test_translate_empty()
    test_translate_mixed_hindi()
    print("\n✓ All translation module tests passed!")
