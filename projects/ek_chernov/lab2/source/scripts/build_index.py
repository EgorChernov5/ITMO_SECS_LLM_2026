"""Разбивает Markdown на чанки и строит dense-эмбеддинги."""

import argparse
import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from source.configuration import get_experiment_config, load_pipeline_config
from source.constants import (
    CONFIG_PATH,
    DEFAULT_CONFIG_NAME,
    INTERIM_DIR,
    MARKDOWN_PATH,
    MODEL_NAME,
    PAGE_MARKER_PATTERN,
    PROJECT_DIR,
)
from source.embedding import DenseEmbedder
from source.structures import Chunk, ExperimentConfig, TextUnit


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры построения индекса.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы командной строки.

    Fallbacks:
        По умолчанию используется baseline и batch size из конфигурации.
    """
    parser = argparse.ArgumentParser(
        description="Построение чанков и эмбеддингов рецептов."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--settings", type=Path, default=CONFIG_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def parse_markdown_units(markdown: str) -> list[TextUnit]:
    """
    Преобразует Markdown в абзацы с номерами страниц и заголовками.

    Parameters:
        markdown (str): Полный Markdown-документ.

    Returns:
        list[TextUnit]: Последовательность текстовых единиц.

    Fallbacks:
        Текст до первого маркера страницы получает номер страницы 0.
    """
    units: list[TextUnit] = []
    headings: list[str] = []
    paragraph: list[str] = []
    page = 0
    in_front_matter = False
    front_matter_finished = False

    def flush() -> None:
        if paragraph:
            text = "\n".join(paragraph).strip()
            if text:
                units.append(TextUnit(text, page, tuple(headings)))
            paragraph.clear()

    for line in markdown.splitlines():
        if line.strip() == "---" and not front_matter_finished:
            if not in_front_matter:
                in_front_matter = True
            else:
                in_front_matter = False
                front_matter_finished = True
            continue
        if in_front_matter:
            continue

        page_match = re.fullmatch(PAGE_MARKER_PATTERN, line.strip())
        if page_match:
            flush()
            page = int(page_match.group(1))
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            headings = headings[: level - 1]
            headings.append(heading_match.group(2).strip())
            continue

        if not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
    flush()
    return units


def render_units(units: list[TextUnit]) -> str:
    """
    Собирает текст чанка с цепочкой заголовков.

    Parameters:
        units (list[TextUnit]): Текстовые единицы одного раздела.

    Returns:
        str: Текст чанка.

    Fallbacks:
        Пустой список возвращает пустую строку.
    """
    if not units:
        return ""
    prefix = "\n".join(
        f"{'#' * min(level, 3)} {heading}"
        for level, heading in enumerate(units[0].headings, start=1)
    )
    body = "\n\n".join(unit.text for unit in units)
    return f"{prefix}\n\n{body}".strip() if prefix else body


def split_long_unit(
    unit: TextUnit,
    experiment: ExperimentConfig,
    embedder: DenseEmbedder,
) -> list[TextUnit]:
    """
    Делит слишком длинный абзац токеновым окном.

    Parameters:
        unit (TextUnit): Исходная текстовая единица.
        experiment (ExperimentConfig): Параметры чанкинга.
        embedder (DenseEmbedder): Эмбеддер с токенизатором.

    Returns:
        list[TextUnit]: Единицы допустимого размера.

    Fallbacks:
        Короткая единица возвращается без изменений.
    """
    heading_text = render_units([TextUnit("", unit.page, unit.headings)])
    heading_size = embedder.token_count(heading_text)
    available = max(1, experiment.chunk_size - heading_size)
    tokens = embedder.encode_tokens(unit.text)
    if len(tokens) <= available:
        return [unit]
    overlap = min(experiment.overlap, available - 1)
    step = max(1, available - overlap)
    return [
        TextUnit(
            embedder.decode_tokens(tokens[start : start + available]),
            unit.page,
            unit.headings,
        )
        for start in range(0, len(tokens), step)
        if tokens[start : start + available]
    ]


def create_chunks(
    units: list[TextUnit],
    experiment: ExperimentConfig,
    embedder: DenseEmbedder,
) -> list[Chunk]:
    """
    Собирает ограниченные по токенам чанки с overlap.

    Parameters:
        units (list[TextUnit]): Единицы исходного документа.
        experiment (ExperimentConfig): Параметры чанкинга.
        embedder (DenseEmbedder): Эмбеддер с токенизатором.

    Returns:
        list[Chunk]: Готовые чанки с детерминированными ID.

    Fallbacks:
        Пустой документ возвращает пустой список.
    """
    expanded = [
        part
        for unit in units
        for part in split_long_unit(unit, experiment, embedder)
    ]
    grouped: list[list[TextUnit]] = []
    current: list[TextUnit] = []

    for unit in expanded:
        # Не смешиваем разные разделы, чтобы заголовок однозначно описывал чанк.
        if current and current[0].headings != unit.headings:
            grouped.append(current)
            current = []
        candidate = [*current, unit]
        if current and embedder.token_count(render_units(candidate)) > (
            experiment.chunk_size
        ):
            grouped.append(current)
            carried: list[TextUnit] = []
            for previous in reversed(current):
                next_carried = [previous, *carried]
                if embedder.token_count(render_units(next_carried)) > (
                    experiment.overlap
                ):
                    break
                carried = next_carried
            current = carried
        current.append(unit)
    if current:
        grouped.append(current)

    chunks: list[Chunk] = []
    for index, group in enumerate(grouped):
        text = render_units(group)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        page_start = min(unit.page for unit in group)
        page_end = max(unit.page for unit in group)
        identifier = uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"{experiment.name}:{index}:{page_start}:{page_end}:"
                f"{content_hash}"
            ),
        )
        chunks.append(
            Chunk(
                id=str(identifier),
                text=text,
                source="Рецепты.pdf",
                source_path="source/data/raw/Рецепты.pdf",
                page_start=page_start,
                page_end=page_end,
                headings=group[0].headings,
                chunk_index=index,
                content_hash=content_hash,
                config_id=experiment.name,
            )
        )
    return chunks


def build_index(
    markdown_path: Path,
    experiment: ExperimentConfig,
    model_name: str,
    batch_size: int,
    rebuild: bool,
) -> Path:
    """
    Создаёт версионированные чанки, эмбеддинги и манифест.

    Parameters:
        markdown_path (Path): Путь к нормализованному Markdown.
        experiment (ExperimentConfig): Параметры эксперимента.
        model_name (str): Идентификатор модели эмбеддингов.
        batch_size (int): Размер GPU-пакета.
        rebuild (bool): Признак принудительной пересборки.

    Returns:
        Path: Путь к каталогу готовой сборки.

    Fallbacks:
        Готовая неизменённая сборка используется повторно.
    """
    if not markdown_path.is_file():
        raise FileNotFoundError(
            f"Markdown не найден: {markdown_path}. Запустите prepare_markdown."
        )
    markdown_bytes = markdown_path.read_bytes()
    build_payload = {
        "markdown_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
        "model": model_name,
        "experiment": asdict(experiment),
    }
    build_hash = hashlib.sha256(
        json.dumps(build_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    build_dir = INTERIM_DIR / experiment.name / build_hash
    manifest_path = build_dir / "manifest.json"
    if manifest_path.is_file() and not rebuild:
        print(f"Используется готовая сборка: {build_dir}")
        return build_dir

    embedder = DenseEmbedder(model_name)
    started = time.perf_counter()
    markdown = markdown_bytes.decode("utf-8")
    chunks = create_chunks(
        parse_markdown_units(markdown),
        experiment,
        embedder,
    )
    chunking_seconds = time.perf_counter() - started
    if not chunks:
        raise ValueError("Не удалось получить ни одного чанка из Markdown.")

    embedding_started = time.perf_counter()
    embeddings = embedder.embed(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
    )
    embedding_seconds = time.perf_counter() - embedding_started

    build_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = build_dir / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(asdict(chunk), ensure_ascii=False) + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
        newline="\n",
    )
    np.save(build_dir / "embeddings.npy", embeddings)
    manifest = {
        **build_payload,
        "build_hash": build_hash,
        "chunk_count": len(chunks),
        "chunking_seconds": chunking_seconds,
        "embedding_seconds": embedding_seconds,
        "batch_size": batch_size,
        "chunks": str(chunks_path.relative_to(PROJECT_DIR).as_posix()),
        "embeddings": str(
            (build_dir / "embeddings.npy").relative_to(PROJECT_DIR).as_posix()
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    latest_path = INTERIM_DIR / experiment.name / "latest.json"
    latest_path.write_text(
        json.dumps(
            {"build_dir": str(build_dir.relative_to(PROJECT_DIR).as_posix())},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return build_dir


def main() -> None:
    """
    Запускает построение чанков и эмбеддингов.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Печатает каталог созданной сборки.

    Fallbacks:
        Размер пакета берётся из конфигурации, если не передан через CLI.
    """
    args = parse_args()
    settings = load_pipeline_config(args.settings)
    experiment = get_experiment_config(settings, args.config)
    embedding = settings.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("Секция embedding должна быть JSON-объектом.")
    model_name = str(embedding.get("model", MODEL_NAME))
    batch_size = args.batch_size or int(embedding.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("Размер пакета должен быть положительным.")
    build_dir = build_index(
        args.markdown,
        experiment,
        model_name,
        batch_size,
        args.rebuild,
    )
    print(f"Индекс построен: {build_dir}")


if __name__ == "__main__":
    main()
