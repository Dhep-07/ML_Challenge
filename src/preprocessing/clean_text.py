"""
Punctuation, special character removal, and capitalization module.
"""
import re

def clean_special_chars_and_capitalize(text: str) -> str:
    """
    Step 3: Remove all characters except word characters (\w), whitespace (\s), and commas (,).
    Collapse repeated whitespace into single spaces and trim.
    Convert to UPPERCASE.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Preserve word characters, whitespace, and commas only
    cleaned = re.sub(r'[^\w\s,]', '', text)
    
    # Collapse repeated whitespace into single spaces and trim
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Convert to uppercase for consistent matching
    return cleaned.upper()

