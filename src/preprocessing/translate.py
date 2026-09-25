
"""
Language translation module using local, offline models only.
Uses argostranslate (MIT licensed, runs fully offline — no external API calls),
which keeps this compliant with the challenge's "no external lookups/APIs" rule.

Install once:
    pip install argostranslate

Download language packages once (also offline after first download):
    import argostranslate.package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    for pkg in available_packages:
        if pkg.from_code != "en":   # any -> en packages
            argostranslate.package.install_from_path(pkg.download())
"""
import re

try:
    import argostranslate.translate
    _installed_languages = argostranslate.translate.get_installed_languages()
    _lang_by_code = {lang.code: lang for lang in _installed_languages}
    _english = _lang_by_code.get("en")
except (ImportError, Exception):
    argostranslate = None
    _installed_languages = []
    _lang_by_code = {}
    _english = None



def _looks_non_english(text: str) -> bool:
    """
    Cheap pre-check: if text is pure ASCII letters/digits/punctuation,
    skip translation entirely (saves time, avoids unnecessary model calls
    on already-English/Latin-script text, which is most of this dataset).
    """
    return bool(re.search(r'[^\x00-\x7F]', text))


def _get_translation(source_lang_code: str):
    if _english is None or source_lang_code not in _lang_by_code:
        return None
    source_lang = _lang_by_code[source_lang_code]
    return source_lang.get_translation(_english)


def translate_to_english(text: str, source_lang_code: str = None) -> str:
    """
    Step 1: Translates input text to English using a local offline model.
    - Returns original text unchanged if it's empty, already English/Latin ASCII,
      or if translation fails for any reason (fail-safe, never drops data).
    - source_lang_code: optional ISO code (e.g. 'hi' for Hindi) if known.
      If not provided, and text is non-ASCII, defaults to attempting
      auto-detected/likely source language packages you've installed.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if not _looks_non_english(text):
        return text  # already Latin/English — skip translation

    try:
        if source_lang_code:
            translation = _get_translation(source_lang_code)
            if translation:
                return translation.translate(text)

        # Fallback: try translating from each installed non-English language
        # until one succeeds cleanly (argostranslate has no built-in
        # language detection, so this is a simple brute-force fallback).
        for lang in _installed_languages:
            if lang.code == "en":
                continue
            translation = lang.get_translation(_english)
            if translation:
                try:
                    result = translation.translate(text)
                    if result and result.strip():
                        return result
                except Exception:
                    continue

        return text  # no suitable package found — return original

    except Exception:
        return text  # fail-safe: never crash the pipeline on translation errors