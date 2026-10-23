"""Проверяет загрузку конфигурации лабораторной работы."""

import pytest

from source.configuration import get_extraction_config, get_experiment_config


def test_extraction_defaults() -> None:
    """
    Проверяет значения OCR по умолчанию.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    extraction = get_extraction_config({})
    assert extraction.use_ocr is False
    assert extraction.ocr_language == "rus"
    assert extraction.ocr_dpi == 300


def test_invalid_ocr_language() -> None:
    """
    Проверяет запрет неподдерживаемого языка OCR.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается ожидаемым исключением.

    Fallbacks:
        Не предусмотрены.
    """
    with pytest.raises(ValueError, match="rus"):
        get_extraction_config(
            {"extraction": {"use_ocr": True, "ocr_language": "eng"}}
        )


def test_experiment_bounds() -> None:
    """
    Проверяет ограничение overlap относительно чанка.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается ожидаемым исключением.

    Fallbacks:
        Не предусмотрены.
    """
    config = {
        "experiments": {
            "invalid": {
                "chunk_size": 100,
                "overlap": 100,
                "hnsw_m": 16,
                "ef_construct": 100,
                "ef_search": 64,
            }
        }
    }
    with pytest.raises(ValueError, match="Overlap"):
        get_experiment_config(config, "invalid")
