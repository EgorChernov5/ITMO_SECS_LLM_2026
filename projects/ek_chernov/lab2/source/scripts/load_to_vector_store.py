"""Загружает локальную сборку эмбеддингов в удалённый Qdrant."""

import argparse
from pathlib import Path

from source.configuration import get_experiment_config, load_pipeline_config
from source.constants import CONFIG_PATH, DEFAULT_CONFIG_NAME, VECTOR_SIZE
from source.vector_store import (
    collection_name,
    create_qdrant_client,
    load_latest_build,
    qdrant_settings,
    wait_for_collection_index,
)


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры загрузки в Qdrant.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы загрузки.

    Fallbacks:
        По умолчанию загружается baseline без удаления коллекции.
    """
    parser = argparse.ArgumentParser(
        description="Загрузка эмбеддингов рецептов в Qdrant."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--settings", type=Path, default=CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--drop-and-reindex", action="store_true")
    parser.add_argument("--index-timeout", type=float)
    return parser.parse_args()


def validate_collection(
    client: object,
    name: str,
    expected_m: int,
    expected_ef_construct: int,
    expected_full_scan_threshold: int,
    expected_indexing_threshold: int,
) -> None:
    """
    Проверяет параметры существующей коллекции.

    Parameters:
        client (object): Клиент Qdrant.
        name (str): Название коллекции.
        expected_m (int): Ожидаемое значение HNSW M.
        expected_ef_construct (int): Ожидаемый ef_construct.
        expected_full_scan_threshold (int): Ожидаемый порог полного перебора.
        expected_indexing_threshold (int): Ожидаемый порог построения индекса.

    Returns:
        None: Завершается при совпадении параметров.

    Fallbacks:
        При несовпадении требует пересоздать коллекцию.
    """
    info = client.get_collection(name)
    hnsw = info.config.hnsw_config
    optimizer = info.config.optimizer_config
    actual = (
        int(hnsw.m),
        int(hnsw.ef_construct),
        int(hnsw.full_scan_threshold),
        int(optimizer.indexing_threshold),
    )
    expected = (
        expected_m,
        expected_ef_construct,
        expected_full_scan_threshold,
        expected_indexing_threshold,
    )
    if actual != expected:
        raise ValueError(
            f"Параметры коллекции {name!r} не совпадают с конфигурацией: "
            f"actual={actual}, expected={expected}. Используйте "
            "--drop-and-reindex."
        )


def upload_build(
    settings: dict[str, object],
    config_name: str,
    batch_size: int,
    drop_and_reindex: bool,
    index_timeout: float | None = None,
) -> int:
    """
    Идемпотентно загружает сборку и ожидает готовности HNSW.

    Parameters:
        settings (dict[str, object]): Конфигурация пайплайна.
        config_name (str): Название эксперимента.
        batch_size (int): Размер пакета upsert.
        drop_and_reindex (bool): Признак пересоздания коллекции.
        index_timeout (float | None): Таймаут индексации из CLI.

    Returns:
        int: Число загруженных и проиндексированных точек.

    Fallbacks:
        При обычном запуске удаляются только устаревшие точки коллекции.
    """
    try:
        from qdrant_client import models
    except ImportError as error:
        raise RuntimeError(
            "Qdrant Client не установлен. Установите requirements.txt."
        ) from error

    experiment = get_experiment_config(settings, config_name)
    _, _, chunks, embeddings = load_latest_build(config_name)
    qdrant = qdrant_settings(settings)
    indexing_threshold = int(qdrant.get("indexing_threshold_kb", 1))
    full_scan_threshold = int(qdrant.get("full_scan_threshold_kb", 0))
    timeout = index_timeout or float(
        qdrant.get("indexing_wait_timeout_seconds", 180)
    )
    client = create_qdrant_client(settings)
    name = collection_name(settings, config_name)

    exists = client.collection_exists(name)
    if exists and drop_and_reindex:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=experiment.hnsw_m,
                ef_construct=experiment.ef_construct,
                full_scan_threshold=full_scan_threshold,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
    else:
        validate_collection(
            client,
            name,
            experiment.hnsw_m,
            experiment.ef_construct,
            full_scan_threshold,
            indexing_threshold,
        )

    current_ids = {str(chunk["id"]) for chunk in chunks}
    for offset in range(0, len(chunks), batch_size):
        points = [
            models.PointStruct(
                id=str(chunk["id"]),
                vector=embeddings[index].tolist(),
                payload=chunk,
            )
            for index, chunk in enumerate(
                chunks[offset : offset + batch_size],
                start=offset,
            )
        ]
        client.upsert(
            collection_name=name,
            points=points,
            wait=True,
        )

    # Удаляем точки предыдущей сборки, не затрагивая актуальные ID.
    stale_ids: list[str] = []
    cursor = None
    while True:
        records, cursor = client.scroll(
            collection_name=name,
            limit=256,
            offset=cursor,
            with_payload=False,
            with_vectors=False,
        )
        stale_ids.extend(
            str(record.id)
            for record in records
            if str(record.id) not in current_ids
        )
        if cursor is None:
            break
    for offset in range(0, len(stale_ids), batch_size):
        client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(
                points=stale_ids[offset : offset + batch_size]
            ),
            wait=True,
        )

    wait_for_collection_index(client, name, len(chunks), timeout)
    return len(chunks)


def main() -> None:
    """
    Запускает загрузку и построение HNSW в Qdrant.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Печатает число проиндексированных точек.

    Fallbacks:
        Не предусмотрены.
    """
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("Размер пакета должен быть положительным.")
    if args.index_timeout is not None and args.index_timeout <= 0:
        raise ValueError("Таймаут индексации должен быть положительным.")
    settings = load_pipeline_config(args.settings)
    count = upload_build(
        settings,
        args.config,
        args.batch_size,
        args.drop_and_reindex,
        args.index_timeout,
    )
    print(
        f"В коллекции {collection_name(settings, args.config)!r} "
        f"построен HNSW для точек: {count}"
    )


if __name__ == "__main__":
    main()
