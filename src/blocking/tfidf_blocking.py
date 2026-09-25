"""
Combined TF-IDF similarity blocking (MAIN RECALL DRIVER).

The backbone of recall for the whole pipeline: doesn't depend on any
specific field being present, works even when addresses are sparse,
unstructured, or landmark-based.

Two complementary TF-IDF runs are performed and unioned:
1. Name-only TF-IDF: char-ngram (3,5) on business_name_normalized
2. Name+Address TF-IDF: char-ngram (3,5) on concatenation of name + address

For each S1 entity, the top-K most similar S2/S3 entities by cosine
similarity are returned as candidates.
"""
import numpy as np
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack as sparse_vstack


def _build_tfidf_pairs(
    s1_texts: List[str],
    other_texts: List[str],
    s1_ids: List[str],
    other_ids: List[str],
    top_k: int = 20,
    threshold: float = 0.15,
    method_label: str = "tfidf",
) -> List[Tuple[str, str, str]]:
    """
    Vectorise both sides with a single TF-IDF vocabulary, then for each
    S1 vector find the top-K most similar S2/S3 vectors above *threshold*.
    """
    if not s1_texts or not other_texts:
        return []

    all_texts = list(other_texts) + list(s1_texts)
    n_other = len(other_texts)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50_000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    other_matrix = tfidf_matrix[:n_other]
    s1_matrix = tfidf_matrix[n_other:]

    pairs: List[Tuple[str, str, str]] = []

    # Process in batches to limit memory
    batch_size = 500
    for start in range(0, s1_matrix.shape[0], batch_size):
        end = min(start + batch_size, s1_matrix.shape[0])
        sim_block = (s1_matrix[start:end] @ other_matrix.T).toarray()

        for i in range(sim_block.shape[0]):
            row = sim_block[i]
            # Get indices above threshold
            above = np.where(row >= threshold)[0]
            if len(above) == 0:
                continue
            # Sort by similarity descending and take top_k
            sorted_idx = above[np.argsort(-row[above])][:top_k]
            s1_id = s1_ids[start + i]
            for j in sorted_idx:
                pairs.append((s1_id, other_ids[j], method_label))

    return pairs


def tfidf_name_blocking(
    df_s1,
    df_other,
    name_col: str = "business_name_normalized",
    top_k: int = 20,
    threshold: float = 0.15,
) -> List[Tuple[str, str, str]]:
    """
    TF-IDF blocking on business name only.
    Returns (s1_id, other_id, "tfidf_name") tuples.
    """
    s1_texts = df_s1[name_col].fillna("").tolist()
    other_texts = df_other[name_col].fillna("").tolist()
    s1_ids = df_s1["entity_id"].tolist()
    other_ids = df_other["entity_id"].tolist()

    return _build_tfidf_pairs(
        s1_texts, other_texts, s1_ids, other_ids,
        top_k=top_k, threshold=threshold,
        method_label="tfidf_name",
    )


def tfidf_combined_blocking(
    df_s1,
    df_other,
    name_col: str = "business_name_normalized",
    address_col: str = "business_address_normalized",
    top_k: int = 20,
    threshold: float = 0.15,
) -> List[Tuple[str, str, str]]:
    """
    TF-IDF blocking on concatenation of name + address.
    Returns (s1_id, other_id, "tfidf_combined") tuples.
    """
    def _concat(row):
        name = str(row.get(name_col, "") or "")
        addr = str(row.get(address_col, "") or "")
        return f"{name} {addr}".strip()

    s1_texts = df_s1.apply(_concat, axis=1).tolist()
    other_texts = df_other.apply(_concat, axis=1).tolist()
    s1_ids = df_s1["entity_id"].tolist()
    other_ids = df_other["entity_id"].tolist()

    return _build_tfidf_pairs(
        s1_texts, other_texts, s1_ids, other_ids,
        top_k=top_k, threshold=threshold,
        method_label="tfidf_combined",
    )
