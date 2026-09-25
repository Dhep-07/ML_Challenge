"""
Blocking package — candidate pair generation for entity resolution.

Strategies (ordered by reliability):
1. Phonetic name code (Metaphone/Soundex)  — PRIMARY
2. Normalized name-prefix                  — SECONDARY
3. City/locality token match               — SUPPLEMENTARY
4. Pincode match                           — OPPORTUNISTIC BONUS
5. TF-IDF char-ngram similarity            — MAIN RECALL DRIVER

Orchestrated by build_candidates.build_candidates().
"""
from src.blocking.build_candidates import build_candidates, evaluate_blocking_recall
from src.blocking.phonetic_blocking import phonetic_blocking, phonetic_key
from src.blocking.name_prefix_blocking import name_prefix_blocking
from src.blocking.city_blocking import city_blocking
from src.blocking.pincode_blocking import pincode_blocking
from src.blocking.tfidf_blocking import tfidf_name_blocking, tfidf_combined_blocking
