"""Содержит загрузку и проверку конфигурации пайплайна."""

import json
from pathlib import Path

from source.constants import CONFIG_PATH
from source.structures import (
    ExperimentConfig,
    ExtractionConfig,
    RetrievalConfig,
    RetrievalModelConfig,
)


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


def get_ollama_context_window(config: dict[str, object]) -> int:
    """
    Возвращает размер контекстного окна Ollama.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.

    Returns:
        int: Размер контекстного окна в токенах.

    Fallbacks:
        Отсутствующее или неположительное значение приводит к ValueError.
    """
    ollama = config.get("ollama")
    if not isinstance(ollama, dict):
        raise ValueError("Секция ollama должна быть JSON-объектом.")
    context_window = int(ollama.get("context_window", 0))
    if context_window <= 0:
        raise ValueError("Параметр context_window должен быть положительным.")
    return context_window


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
    if "splitter" not in raw:
        raise ValueError(
            f"В конфигурации {name!r} отсутствует обязательное поле splitter."
        )
    splitter = str(raw["splitter"])
    if splitter not in {"markdown_header", "long_late_chunking"}:
        raise ValueError(
            "Поле splitter должно быть markdown_header или long_late_chunking."
        )
    if "retrieval" not in raw:
        raise ValueError(
            f"В конфигурации {name!r} отсутствует обязательное поле retrieval."
        )
    retrieval = str(raw["retrieval"])
    get_retrieval_config(config, retrieval)
    experiment = ExperimentConfig(
        name=name,
        splitter=splitter,
        chunk_size=int(raw["chunk_size"]),
        overlap=int(raw["overlap"]),
        hnsw_m=int(raw["hnsw_m"]),
        ef_construct=int(raw["ef_construct"]),
        ef_search=int(raw["ef_search"]),
        retrieval=retrieval,
    )
    if experiment.chunk_size <= 0:
        raise ValueError("Размер чанка должен быть положительным.")
    if not 0 <= experiment.overlap < experiment.chunk_size:
        raise ValueError("Overlap должен быть меньше размера чанка.")
    if (
        experiment.splitter == "long_late_chunking"
        and experiment.chunk_size > 510
    ):
        raise ValueError(
            "Макроокно Long Late Chunking не должно превышать 510 "
            "содержательных токенов SBERT."
        )
    return experiment


def get_retrieval_model_config(
    config: dict[str, object],
    name: str,
    expected_type: str | None = None,
) -> RetrievalModelConfig:
    """
    Возвращает проверенную конфигурацию модели retrieval.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.
        name (str): Имя модели.
        expected_type (str | None): Ожидаемый тип модели. По умолчанию: None.

    Returns:
        RetrievalModelConfig: Проверенные параметры модели.

    Fallbacks:
        Отсутствующая модель и неизвестный тип приводят к ValueError.
    """
    models = config.get("retrieval_models")
    if not isinstance(models, dict) or name not in models:
        raise ValueError(f"Модель retrieval {name!r} не найдена.")
    raw = models[name]
    if not isinstance(raw, dict):
        raise ValueError(f"Модель retrieval {name!r} должна быть JSON-объектом.")
    model_type = str(raw.get("type", ""))
    if model_type not in {"bi_encoder", "cross_encoder"}:
        raise ValueError(f"Неизвестный тип модели retrieval: {model_type!r}.")
    if expected_type is not None and model_type != expected_type:
        raise ValueError(
            f"Модель {name!r} должна иметь тип {expected_type}, "
            f"получен {model_type}."
        )
    model = str(raw.get("model", "")).strip()
    if not model:
        raise ValueError(f"У модели {name!r} отсутствует обязательное поле model.")
    batch_size = int(raw.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError(f"batch_size модели {name!r} должен быть положительным.")
    return RetrievalModelConfig(name, model_type, model, batch_size)


def get_retrieval_config(
    config: dict[str, object],
    name: str,
) -> RetrievalConfig:
    """
    Разрешает ссылки и возвращает готовую retrieval-конфигурацию.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.
        name (str): Имя retrieval-конфигурации.

    Returns:
        RetrievalConfig: Полностью разрешённая конфигурация.

    Fallbacks:
        Некорректная ссылка или параметр приводят к ValueError без fallback.
    """
    retrievals = config.get("retrievals")
    if not isinstance(retrievals, dict) or name not in retrievals:
        raise ValueError(f"Retrieval-конфигурация {name!r} не найдена.")
    raw = retrievals[name]
    if not isinstance(raw, dict):
        raise ValueError(f"Retrieval {name!r} должен быть JSON-объектом.")
    retrieval_type = str(raw.get("type", ""))
    if retrieval_type not in {"bi", "bi_n_cross"}:
        raise ValueError(f"Неизвестный тип retrieval: {retrieval_type!r}.")
    if "bi_encoder" not in raw:
        raise ValueError(f"У retrieval {name!r} отсутствует bi_encoder.")
    bi_encoder = get_retrieval_model_config(
        config,
        str(raw["bi_encoder"]),
        "bi_encoder",
    )
    top_k = int(raw.get("top_k", 0))
    if top_k <= 0:
        raise ValueError(f"top_k retrieval {name!r} должен быть положительным.")
    cross_encoder = None
    candidate_top_k = None
    if retrieval_type == "bi_n_cross":
        if "cross_encoder" not in raw:
            raise ValueError(f"У retrieval {name!r} отсутствует cross_encoder.")
        cross_encoder = get_retrieval_model_config(
            config,
            str(raw["cross_encoder"]),
            "cross_encoder",
        )
        candidate_top_k = int(raw.get("candidate_top_k", 0))
        if candidate_top_k <= 0:
            raise ValueError(
                f"candidate_top_k retrieval {name!r} должен быть положительным."
            )
        if candidate_top_k < top_k:
            raise ValueError("candidate_top_k должен быть не меньше top_k.")
    return RetrievalConfig(
        name,
        retrieval_type,
        bi_encoder,
        cross_encoder,
        candidate_top_k,
        top_k,
    )


def get_experiment_retrieval_config(
    config: dict[str, object],
    experiment: ExperimentConfig,
) -> RetrievalConfig:
    """
    Возвращает retrieval, связанный с проверенным экспериментом.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.
        experiment (ExperimentConfig): Проверенный эксперимент.

    Returns:
        RetrievalConfig: Разрешённая retrieval-конфигурация.

    Fallbacks:
        Ошибки ссылок передаются из get_retrieval_config.
    """
    return get_retrieval_config(config, experiment.retrieval)
