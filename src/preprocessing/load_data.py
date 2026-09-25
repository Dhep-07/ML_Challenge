"""
Data loader utilities for TSV files (sep="\\t").
"""
import pandas as pd

def load_tsv(file_path: str) -> pd.DataFrame:
    """Loads TSV dataset with tab separation."""
    return pd.read_csv(file_path, sep="\t")
