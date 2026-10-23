"""Содержит общие операции с удалённым Qdrant."""

import json
import os
import time
from pathlib import Path

import numpy as np

from source.constants import INTERIM_DIR, PROJECT_DIR
from source.structures import ExperimentConfig


def load_latest_build(
    config_name: str,
) -> tuple[Path, dict[str, object], list[dict[str, object]], object]:
    """
    Загружает последнюю локальную сборку выбранной конфигурации.

    Parameters:
        config_name (str): Название экспериментальной конфигурации.

    Returns:
        tuple[Path, dict[str, object], list[dict[str, object]], object]:
            Каталог, манифест, чанки и матрица эмбеддингов.

    Fallbacks:
        При отсутствии сборки возбуждает FileNotFoundError.
    """
    latest_path = INTERIM_DIR / config_name / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(
            f"Сборка {config_name!r} не найдена. Запустите build_index."
        )
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    build_dir = PROJECT_DIR / str(latest["build_dir"])
    manifest = json.loads(
        (build_dir / "manifest.json").read_text(encoding="utf-8")
    )
    chunks = [
        json.loads(line)
        for line in (build_dir / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    embeddings = np.load(build_dir / "embeddings.npy")
    if len(chunks) != len(embeddings):
        raise ValueError("Число чанков не совпадает с числом эмбеддингов.")
    return build_dir, manifest, chunks, embeddings


def qdrant_settings(config: dict[str, object]) -> dict[str, object]:
    """
    Возвращает проверенную секцию Qdrant.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.

    Returns:
        dict[str, object]: Параметры подключения и индексации.

    Fallbacks:
        При отсутствии секции возвращает пустой объект.
    """
    qdrant = config.get("qdrant", {})
    if not isinstance(qdrant, dict):
        raise ValueError("Секция qdrant должна быть JSON-объектом.")
    return qdrant


def qdrant_url(config: dict[str, object]) -> str:
    """
    Возвращает адрес Qdrant с учётом переменной окружения.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.

    Returns:
        str: URL HTTP API Qdrant.

    Fallbacks:
        Использует http://localhost:6333 при отсутствии настройки.
    """
    qdrant = qdrant_settings(config)
    return os.getenv(
        "QDRANT_URL",
        str(qdrant.get("url", "http://localhost:6333")),
    )


def create_qdrant_client(config: dict[str, object]) -> object:
    """
    Создаёт клиент удалённого Qdrant.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.

    Returns:
        object: Настроенный QdrantClient.

    Fallbacks:
        API key не используется, если QDRANT_API_KEY отсутствует.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError(
            "Qdrant Client не установлен. Установите requirements.txt."
        ) from error

    qdrant = qdrant_settings(config)
    return QdrantClient(
        url=qdrant_url(config),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=float(qdrant.get("request_timeout_seconds", 30)),
    )


def collection_name(config: dict[str, object], config_name: str) -> str:
    """
    Формирует имя коллекции Qdrant.

    Parameters:
        config (dict[str, object]): Полная конфигурация пайплайна.
        config_name (str): Название эксперимента.

    Returns:
        str: Имя коллекции.

    Fallbacks:
        Использует префикс recipes при отсутствии настройки.
    """
    prefix = str(qdrant_settings(config).get("collection_prefix", "recipes"))
    return f"{prefix}_{config_name}"


def wait_for_collection_index(
    client: object,
    name: str,
    expected_vectors: int,
    timeout_seconds: float,
) -> None:
    """
    Ожидает готовности HNSW-индекса коллекции.

    Parameters:
        client (object): Клиент Qdrant.
        name (str): Название коллекции.
        expected_vectors (int): Ожидаемое число проиндексированных векторов.
        timeout_seconds (float): Максимальное время ожидания.

    Returns:
        None: Завершается после полной индексации.

    Fallbacks:
        По истечении таймаута возбуждает TimeoutError с текущим состоянием.
    """
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"
    last_indexed = 0
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        last_status = str(
            getattr(info.status, "value", info.status)
        ).lower()
        last_indexed = int(info.indexed_vectors_count or 0)
        if last_status == "green" and last_indexed >= expected_vectors:
            return
        time.sleep(1)
    raise TimeoutError(
        f"Коллекция {name!r} не построила HNSW за {timeout_seconds:.0f} с: "
        f"status={last_status}, indexed={last_indexed}, "
        f"expected={expected_vectors}."
    )


def search_chunks(
    client: object,
    name: str,
    vector: list[float],
    limit: int,
    ef_search: int,
) -> list[dict[str, object]]:
    """
    Ищет ближайшие чанки в Qdrant.

    Parameters:
        client (object): Клиент Qdrant.
        name (str): Название коллекции.
        vector (list[float]): Вектор поискового запроса.
        limit (int): Число результатов.
        ef_search (int): Параметр точности поиска HNSW.

    Returns:
        list[dict[str, object]]: Результаты с ID, score и payload.

    Fallbacks:
        Ошибки подключения передаются вызывающему коду.
    """
    from qdrant_client import models

    response = client.query_points(
        collection_name=name,
        query=vector,
        limit=limit,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=ef_search, exact=False),
    )
    return [
        {
            "id": str(point.id),
            "score": float(point.score),
            "payload": point.payload or {},
        }
        for point in response.points
    ]
