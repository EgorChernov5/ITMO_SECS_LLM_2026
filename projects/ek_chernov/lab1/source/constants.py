"""Содержит статические настройки сценариев лабораторной работы."""

OLLAMA_URL = "http://localhost:11434"
OPENAI_BASE_URL = f"{OLLAMA_URL}/v1"

MODELS = (
    "qwen2.5:3b-instruct",
    "llama3.2:3b",
    "ministral-3:3b",
)

PROMPTS = {
    "P1_generation": "text",
    "P2_classification": "text",
    "P3_extraction": "text",
}

MODES: dict[str, dict[str, int | float]] = {
    "A_baseline": {},
    "B_tuned": {
        "temperature": 0.2,
        "max_tokens": 256,
    },
}

# Трёх повторов достаточно для базовой оценки воспроизводимости.
REPEATS = 3

RESULT_FIELDS = (
    "model",
    "task",
    "mode",
    "repeat",
    "time_seconds",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
    "finish_reason",
    "error",
)

METRICS = {
    "time_seconds": ("Время HTTP-запроса", "сек"),
    "completion_tokens": ("Выходные токены", "токены"),
    "prompt_tokens": ("Входные токены", "токены"),
    "tokens_per_second": ("Скорость по полному времени", "токенов/с"),
}
