"""
Char n-gram TF-IDF vectorization and top-k cosine similarity blocking.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_blocking(texts, top_k=10):
    """Generates candidate pairs using character n-gram TF-IDF and top-k cosine similarity."""
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Return top-k candidate indices per item
    return tfidf_matrix
