#!/usr/bin/env python3
"""
Setup script for ML-Challenge.
1. Ensures required dependencies are installed.
2. Downloads and installs Argos Translate offline translation packages.
"""

import sys
import subprocess

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def install_packages():
    print("=== Step 1: Installing dependencies ===")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_argostranslate():
    print("\n=== Step 2: Setting up Argos Translate Models ===")
    import argostranslate.package

    print("Updating package index...")
    argostranslate.package.update_package_index()
    
    available_packages = argostranslate.package.get_available_packages()
    print(f"Found {len(available_packages)} available packages.")

    # Install packages where target language is English
    installed_count = 0
    for pkg in available_packages:
        if pkg.to_code == "en":
            print(f"Downloading & installing package: {pkg.from_code} -> {pkg.to_code} ({pkg.from_name} to {pkg.to_name})")
            download_path = pkg.download()
            argostranslate.package.install_from_path(download_path)
            installed_count += 1

    print(f"Successfully installed {installed_count} translation package(s).")

if __name__ == "__main__":
    try:
        install_packages()
        setup_argostranslate()
        print("\n=== Setup completed successfully! ===")
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)
