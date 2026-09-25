"""
Pipeline coordinator for the 3-step preprocessing workflow:
1. Translate to English (local, offline argostranslate)
2. Expand domain-specific abbreviations (field-aware: name vs address)
3. Capitalize (UPPERCASE) while preserving special characters and formatting
"""
from typing import Union, Dict, Any
import pandas as pd
from src.preprocessing.translate import translate_to_english
from src.preprocessing.expand_abbreviations import expand_abbreviations
from src.preprocessing.clean_text import clean_special_chars_and_capitalize

def preprocess_text(text: str, field_type: str = 'name') -> str:
    """
    Executes the 3-step sequential preprocessing pipeline on a single text value.
    
    Step 1: Translate to English (offline argostranslate, fail-safe)
    Step 2: Expand abbreviations (field-specific)
    Step 3: Normalize whitespace & capitalize (preserving special characters)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        # Step 1: Translate to English (offline model)
        translated = translate_to_english(text)
        
        # Step 2: Lowercase & expand abbreviations
        expanded = expand_abbreviations(translated, field_type=field_type)
        
        # Step 3: Strip special characters & uppercase
        cleaned = clean_special_chars_and_capitalize(expanded)
        
        return cleaned
    except Exception:
        # End-to-end fail-safe guarantee
        return str(text).upper().strip() if text else ""

def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single record/dictionary and adds normalized columns.
    """
    record = record.copy()
    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    
    record["business_name_normalized"] = preprocess_text(raw_name, field_type='name')
    record["business_address_normalized"] = preprocess_text(raw_address, field_type='address')
    
    return record

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a pandas DataFrame (TSV format) in place or returned, adding:
    - business_name_normalized
    - business_address_normalized
    leaving raw original columns untouched.
    """
    df = df.copy()
    
    if "business_name" in df.columns:
        df["business_name_normalized"] = df["business_name"].apply(
            lambda x: preprocess_text(str(x) if pd.notna(x) else "", field_type='name')
        )
    else:
        df["business_name_normalized"] = ""

    if "business_address" in df.columns:
        df["business_address_normalized"] = df["business_address"].apply(
            lambda x: preprocess_text(str(x) if pd.notna(x) else "", field_type='address')
        )
    else:
        df["business_address_normalized"] = ""

    return df

