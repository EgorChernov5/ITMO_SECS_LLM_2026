"""Содержит загрузку и проверку конфигурации пайплайна."""

import json
from pathlib import Path

from source.constants import CONFIG_PATH
from source.structures import ExperimentConfig, ExtractionConfig


def load_pipeline_config(
    path: Path = CONFIG_PATH,
) -> dict[str, object]:
    """
    Загружает JSON-конфигурацию пайплайна.

    Parameters:
        path (Path): Путь к конфигурации. По умолчанию: CONFIG_PATH.

    Returns:
        dict[str, object]: Проверенная корневая секция конфигурации.

    Fallbacks:
        При ошибке формата или отсутствии объекта верхнего уровня возбуждает ValueError.
    """
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Не удалось прочитать конфигурацию {path}: {error}"
        raise ValueError(message) from error
    if not isinstance(config, dict):
        raise ValueError("Корень конфигурации должен быть JSON-объектом.")
    return config


def get_extraction_config(config: dict[str, object]) -> ExtractionConfig:
    """
    Возвращает параметры извлечения текста.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.

    Returns:
        ExtractionConfig: Проверенные параметры извлечения.

    Fallbacks:
        При отсутствии секции использует значения ExtractionConfig по умолчанию.
    """
    raw = config.get("extraction", {})
    if not isinstance(raw, dict):
        raise ValueError("Секция extraction должна быть JSON-объектом.")
    extraction = ExtractionConfig(
        use_ocr=bool(raw.get("use_ocr", False)),
        ocr_language=str(raw.get("ocr_language", "rus")),
        ocr_dpi=int(raw.get("ocr_dpi", 300)),
    )
    if extraction.ocr_language != "rus":
        raise ValueError("Для OCR поддерживается языковая конфигурация rus.")
    if extraction.ocr_dpi <= 0:
        raise ValueError("Параметр ocr_dpi должен быть положительным.")
    return extraction


def get_experiment_config(
    config: dict[str, object],
    name: str,
) -> ExperimentConfig:
    """
    Возвращает параметры выбранного эксперимента.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.
        name (str): Название эксперимента.

    Returns:
        ExperimentConfig: Проверенные параметры чанкинга и HNSW.

    Fallbacks:
        При отсутствии эксперимента или некорректных значениях возбуждает ValueError.
    """
    experiments = config.get("experiments")
    if not isinstance(experiments, dict) or name not in experiments:
        available = ", ".join(experiments) if isinstance(experiments, dict) else ""
        raise ValueError(
            f"Конфигурация {name!r} не найдена. Доступны: {available}."
        )
    raw = experiments[name]
    if not isinstance(raw, dict):
        raise ValueError(f"Конфигурация {name!r} должна быть JSON-объектом.")
    experiment = ExperimentConfig(
        name=name,
        chunk_size=int(raw["chunk_size"]),
        overlap=int(raw["overlap"]),
        hnsw_m=int(raw["hnsw_m"]),
        ef_construct=int(raw["ef_construct"]),
        ef_search=int(raw["ef_search"]),
    )
    if not 100 <= experiment.chunk_size <= 1000:
        raise ValueError("Размер чанка должен находиться в диапазоне 100–1000.")
    if not 0 <= experiment.overlap < experiment.chunk_size:
        raise ValueError("Overlap должен быть меньше размера чанка.")
    return experiment
