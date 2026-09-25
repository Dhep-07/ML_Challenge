"""
Configuration settings for paths, thresholds, and hyperparameters.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "student_resource"
OUTPUT_DIR = BASE_DIR / "output"

# Thresholds & Hyperparameters
SIMILARITY_THRESHOLD = 0.5
DEFAULT_COSINE_TOP_K = 10
F_BETA = 0.5
