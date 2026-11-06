"""Содержит пути и статические параметры лабораторной работы."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "source"
CONFIG_PATH = SOURCE_DIR / "config" / "pipeline.json"
RAW_PDF_PATH = SOURCE_DIR / "data" / "raw" / "Рецепты.pdf"
MARKDOWN_PATH = SOURCE_DIR / "data" / "processed" / "recipes.md"
RECIPES_PATH = SOURCE_DIR / "data" / "processed" / "recipes_2.md"
EXPLORED_DIR = SOURCE_DIR / "data" / "explored"
INTERIM_DIR = SOURCE_DIR / "data" / "interim"
RESULTS_DIR = SOURCE_DIR / "data" / "results"
RESULTS_RUNS_DIR = RESULTS_DIR / "runs"
QUESTIONS_PATH = SOURCE_DIR / "evaluation" / "questions.json"
MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
QWEN_TOKENIZER_NAME = "Qwen/Qwen2.5-3B-Instruct"
GENERATION_RESPONSE_TOKEN_RESERVE = 1024
PROMPT_TOKEN_SAFETY_MARGIN = 128
PROMPT_CONTEXT_OVERFLOW_MESSAGE = (
    "Попробуйте спросить что-то одно. Я не могу всё скушать"
)
VECTOR_SIZE = 1024
PAGE_MARKER_PATTERN = r"<!--\s*page:\s*(\d+)\s*-->"
DEFAULT_CONFIG_NAME = "baseline"
INVALID_EMPTY_QUESTION_RESPONSE = (
    "Запрос невалиден: запрос не должен быть пустым."
)
INVALID_LONG_QUESTION_RESPONSE = (
    "Запрос невалиден: длина запроса составляет {token_count} токенов, "
    "а контекстное окно модели допускает не более {max_tokens} токенов."
)
