"""
Punctuation, special character removal, and capitalization module.
"""
import re

def clean_special_chars_and_capitalize(text: str) -> str:
    """
    Step 3: Collapse repeated whitespace into single spaces and trim.
    Convert to UPPERCASE (preserving all special characters).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Collapse repeated whitespace into single spaces and trim
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to uppercase for consistent matching
    return cleaned.upper()


