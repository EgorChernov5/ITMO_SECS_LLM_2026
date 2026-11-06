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
                "splitter": "markdown_header",
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


@pytest.mark.parametrize("splitter", ("markdown_header", "long_late_chunking"))
def test_supported_splitters(splitter: str) -> None:
    """
    Проверяет оба поддерживаемых значения splitter.

    Parameters:
        splitter (str): Проверяемая стратегия.

    Returns:
        None: Проверка завершается утверждением pytest.

    Fallbacks:
        Не предусмотрены.
    """
    # Собираем минимальную корректную конфигурацию эксперимента.
    config = {
        "experiments": {
            "valid": {
                "splitter": splitter,
                "chunk_size": 100,
                "overlap": 10,
                "hnsw_m": 16,
                "ef_construct": 100,
                "ef_search": 64,
            }
        }
    }
    assert get_experiment_config(config, "valid").splitter == splitter


@pytest.mark.parametrize(
    ("raw", "message"),
    (({}, "отсутствует"), ({"splitter": "unknown"}, "splitter")),
)
def test_invalid_splitter(raw: dict[str, object], message: str) -> None:
    """
    Проверяет обязательность и допустимые значения splitter.

    Parameters:
        raw (dict[str, object]): Поля некорректного эксперимента.
        message (str): Ожидаемый фрагмент ошибки.

    Returns:
        None: Проверка завершается ожидаемым исключением.

    Fallbacks:
        Остальные поля не требуются, потому что splitter проверяется первым.
    """
    # Проверяем ошибку до чтения остальных обязательных параметров.
    with pytest.raises(ValueError, match=message):
        get_experiment_config({"experiments": {"invalid": raw}}, "invalid")


def test_long_late_chunking_context_limit() -> None:
    """
    Проверяет раннее отклонение слишком большого макроокна.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается ожидаемым исключением.

    Fallbacks:
        Фактический лимит дополнительно проверяется после загрузки токенизатора.
    """
    # Передаём окно, которое не помещается в стандартный контекст SBERT.
    config = {
        "experiments": {
            "invalid": {
                "splitter": "long_late_chunking",
                "chunk_size": 511,
                "overlap": 10,
                "hnsw_m": 16,
                "ef_construct": 100,
                "ef_search": 64,
            }
        }
    }
    with pytest.raises(ValueError, match="Макроокно"):
        get_experiment_config(config, "invalid")
