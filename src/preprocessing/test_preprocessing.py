"""
Unit tests for the preprocessing module.
"""
import pandas as pd
from src.preprocessing import preprocess_text, preprocess_dataframe, preprocess_record

def test_step1_ascii_passthrough():
    text = "Acme Inc 123 Main"
    # Pure ASCII should skip translation and pass through
    processed = preprocess_text(text, field_type='name')
    assert processed == "ACME INCORPORATED 123 MAIN"


def test_step2_abbreviations_name():
    raw_name = "Acme Pvt. Ltd. & Co. Intl."
    normalized = preprocess_text(raw_name, field_type='name')
    assert normalized == "ACME PRIVATE LIMITED AND COMPANY INTERNATIONAL"

def test_step2_abbreviations_address():
    raw_addr = "123 Main Rd. Apt 4B, Bldg 2, Opp Sec 5"
    normalized = preprocess_text(raw_addr, field_type='address')
    assert normalized == "123 MAIN ROAD APARTMENT 4B, BUILDING 2, OPPOSITE SECTOR 5"

def test_step3_commas_preserved():
    raw_addr = "Suite 100, 456 Park Ave, New York, NY"
    normalized = preprocess_text(raw_addr, field_type='address')
    assert "," in normalized
    assert normalized == "SUITE 100, 456 PARK AVENUE, NEW YORK, NY"

def test_dataframe_preprocessing():
    data = {
        "entity_id": [1, 2],
        "business_name": ["Acme Pvt. Ltd.", "Global Mfg Corp"],
        "business_address": ["12 High Rd.", "45 Industrial Blvd"],
        "country": ["IN", "US"]
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_dataframe(df)

    assert "business_name_normalized" in processed_df.columns
    assert "business_address_normalized" in processed_df.columns
    assert processed_df["business_name_normalized"].iloc[0] == "ACME PRIVATE LIMITED"
    assert processed_df["business_address_normalized"].iloc[0] == "12 HIGH ROAD"
    # Ensure original columns were left untouched
    assert processed_df["business_name"].iloc[0] == "Acme Pvt. Ltd."

def test_failsafe_empty_nan():
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""

if __name__ == "__main__":
    test_step1_ascii_passthrough()
    test_step2_abbreviations_name()
    test_step2_abbreviations_address()
    test_step3_commas_preserved()
    test_dataframe_preprocessing()
    test_failsafe_empty_nan()
    print("All preprocessing module tests passed!")
