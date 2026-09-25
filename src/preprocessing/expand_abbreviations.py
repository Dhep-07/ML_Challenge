"""
Abbreviation expansion module.
Expands domain-specific business name and address abbreviations.
Must run before punctuation stripping (preserves markers like pvt. or &).
"""
import re

# Abbreviation dictionaries (lowercase keys for regex word boundary replacement)
NAME_ABBREVIATIONS = {
    r'\bpvt\b\.?': 'private',
    r'\bltd\b\.?': 'limited',
    r'\bcorp\b\.?': 'corporation',
    r'\bco\b\.?': 'company',
    r'\binc\b\.?': 'incorporated',
    r'\bllc\b\.?': 'limited liability company',
    r'\bllp\b\.?': 'limited liability partnership',
    r'\bintl\b\.?': 'international',
    r'\bmfg\b\.?': 'manufacturing',
    r'\bsvcs\b\.?': 'services',
    r'\bsvc\b\.?': 'service',
    r'\bdept\b\.?': 'department',
    r'\bassoc\b\.?': 'associates',
    r'\bbros\b\.?': 'brothers',
    r'&': 'and',
}


ADDRESS_ABBREVIATIONS = {
    r'\brd\b\.?': 'road',
    r'\bst\b\.?': 'street',
    r'\bave\b\.?': 'avenue',
    r'\bblvd\b\.?': 'boulevard',
    r'\bapt\b\.?': 'apartment',
    r'\bfl\b\.?': 'floor',
    r'\bnr\b\.?': 'near',
    r'\bopp\b\.?': 'opposite',
    r'\bbldg\b\.?': 'building',
    r'\bln\b\.?': 'lane',
    r'\bdr\b\.?': 'drive',
    r'\bhwy\b\.?': 'highway',
    r'\bsec\b\.?': 'sector',
}

def expand_abbreviations(text: str, field_type: str = 'name') -> str:
    """
    Step 2: Lowercases the text and applies domain-specific abbreviation expansion.
    
    Args:
        text: Input string.
        field_type: 'name' for business names or 'address' for business addresses.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase text first as specified in Step 2
    result = text.lower()
    
    abbrev_dict = NAME_ABBREVIATIONS if field_type == 'name' else ADDRESS_ABBREVIATIONS
    
    for pattern, replacement in abbrev_dict.items():
        result = re.sub(pattern, replacement, result)

    return result

