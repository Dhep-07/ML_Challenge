#!/usr/bin/env python3
"""
Setup script for ML-Challenge.
1. Ensures required dependencies are installed.
2. Downloads and installs Argos Translate offline translation packages (fallback).
3. Pre-downloads IndicTrans2 (AI4Bharat, MIT) for Indian languages.
4. Pre-downloads Opus-MT (Helsinki-NLP, CC-BY-4.0) for French + key languages.
"""

import sys
import subprocess

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def install_packages():
    print("=== Step 1: Installing pip dependencies ===")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_argostranslate():
    print("\n=== Step 2: Setting up Argos Translate Models (fallback) ===")
    import argostranslate.package

    print("Updating package index...")
    argostranslate.package.update_package_index()

    available_packages = argostranslate.package.get_available_packages()
    print(f"Found {len(available_packages)} available packages.")

    # Install packages where target language is English
    installed_count = 0
    for pkg in available_packages:
        if pkg.to_code == "en":
            print(f"Downloading & installing: {pkg.from_code} -> en ({pkg.from_name})")
            download_path = pkg.download()
            argostranslate.package.install_from_path(download_path)
            installed_count += 1

    print(f"Successfully installed {installed_count} Argos Translate package(s).")

def setup_indictrans2():
    print("\n=== Step 3: Pre-downloading IndicTrans2 (AI4Bharat, MIT License) ===")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        print(f"Downloading model: {model_name}")
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        print("IndicTrans2 model cached successfully.")
    except Exception as e:
        print(f"[WARNING] IndicTrans2 download failed (non-fatal): {e}")
        print("  Indian language translation will fall back to Argos Translate.")

def setup_opus_mt():
    print("\n=== Step 4: Pre-downloading Opus-MT models (Helsinki-NLP, CC-BY-4.0) ===")
    # Pre-download key language models — French is especially important
    # since it appears in the test set
    key_models = {
        "fr": "Helsinki-NLP/opus-mt-fr-en",
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "de": "Helsinki-NLP/opus-mt-de-en",
        "es": "Helsinki-NLP/opus-mt-es-en",
    }
    try:
        from transformers import MarianMTModel, MarianTokenizer

        for lang, model_name in key_models.items():
            try:
                print(f"Downloading: {model_name}")
                MarianTokenizer.from_pretrained(model_name)
                MarianMTModel.from_pretrained(model_name)
                print(f"  ✓ {lang}→en cached.")
            except Exception as e:
                print(f"  [WARNING] {model_name} failed (non-fatal): {e}")
    except ImportError as e:
        print(f"[WARNING] transformers not available: {e}")

if __name__ == "__main__":
    try:
        install_packages()
        setup_argostranslate()
        setup_indictrans2()
        setup_opus_mt()
        print("\n=== Setup completed successfully! ===")
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)
