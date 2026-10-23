"""Содержит пути и статические параметры лабораторной работы."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "source"
CONFIG_PATH = SOURCE_DIR / "config" / "pipeline.json"
RAW_PDF_PATH = SOURCE_DIR / "data" / "raw" / "Рецепты.pdf"
MARKDOWN_PATH = SOURCE_DIR / "data" / "processed" / "recipes.md"
INTERIM_DIR = SOURCE_DIR / "data" / "interim"
RESULTS_DIR = SOURCE_DIR / "data" / "results"
QUESTIONS_PATH = SOURCE_DIR / "evaluation" / "questions.json"
MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
VECTOR_SIZE = 1024
PAGE_MARKER_PATTERN = r"<!--\s*page:\s*(\d+)\s*-->"
DEFAULT_CONFIG_NAME = "baseline"
