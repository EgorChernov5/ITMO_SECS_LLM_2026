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


class IndexingStrategy:
    """
    Задаёт общий контракт построения чанков и эмбеддингов.

    Attributes:
        experiment (ExperimentConfig): Параметры выбранного эксперимента.
        embedder (DenseEmbedder): Загруженный SBERT-эмбеддер.
        batch_size (int): Размер GPU-пакета.
        macro_window_count (int): Число обработанных макроокон. По умолчанию: 0.

    Fallbacks:
        Нереализованные этапы стратегии возбуждают NotImplementedError.
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        embedder: DenseEmbedder,
        batch_size: int,
    ) -> None:
        """
        Сохраняет общие зависимости стратегии.

        Parameters:
            experiment (ExperimentConfig): Параметры эксперимента.
            embedder (DenseEmbedder): Загруженный SBERT-эмбеддер.
            batch_size (int): Размер GPU-пакета.

        Returns:
            None: Создаёт стратегию.

        Fallbacks:
            Проверка параметров выполняется конфигурационным слоем.
        """
        self.experiment = experiment
        self.embedder = embedder
        self.batch_size = batch_size
        self.macro_window_count = 0
        self.chunking_seconds = 0.0
        self.embedding_seconds = 0.0

    @staticmethod
    def make_chunk(
        text: str,
        units: list[TextUnit],
        index: int,
        experiment: ExperimentConfig,
        headings: tuple[str, ...] | None = None,
    ) -> Chunk:
        """
        Формирует общие метаданные и детерминированный ID чанка.

        Parameters:
            text (str): Полный текст физического чанка.
            units (list[TextUnit]): Единицы с заголовками и страницами.
            index (int): Порядковый номер чанка.
            experiment (ExperimentConfig): Параметры эксперимента.
            headings (tuple[str, ...] | None): Явные заголовки чанка. По умолчанию: None.

        Returns:
            Chunk: Чанк с заполненными общими полями.

        Fallbacks:
            Пустой список единиц не поддерживается.
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        page_start = min(unit.page for unit in units)
        page_end = max(unit.page for unit in units)
        identifier = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{experiment.name}:{index}:{page_start}:{page_end}:{content_hash}",
        )
        return Chunk(
            id=str(identifier),
            text=text,
            source="Рецепты.pdf",
            source_path="source/data/raw/Рецепты.pdf",
            page_start=page_start,
            page_end=page_end,
            headings=headings if headings is not None else units[0].headings,
            chunk_index=index,
            content_hash=content_hash,
            config_id=experiment.name,
        )

    def create_chunks(self, markdown: str) -> list[Chunk]:
        """
        Выделяет физические чанки выбранной стратегией.

        Parameters:
            markdown (str): Нормализованный Markdown.

        Returns:
            list[Chunk]: Физические чанки в порядке документа.

        Fallbacks:
            Базовый класс требует переопределения.
        """
        raise NotImplementedError

    def create_embeddings(
        self,
        markdown: str,
        chunks: list[Chunk],
    ) -> object:
        """
        Строит векторы, соответствующие физическим чанкам.

        Parameters:
            markdown (str): Нормализованный Markdown.
            chunks (list[Chunk]): Физические чанки.

        Returns:
            object: Матрица NumPy float32.

        Fallbacks:
            Базовый класс требует переопределения.
        """
        raise NotImplementedError

    def build(self, markdown: str) -> tuple[list[Chunk], object]:
        """
        Выполняет оба этапа стратегии и проверяет общий контракт.

        Parameters:
            markdown (str): Нормализованный Markdown.

        Returns:
            tuple[list[Chunk], object]: Чанки и матрица эмбеддингов.

        Fallbacks:
            Пустой результат и несовпадающие размерности отклоняются.
        """
        # Отдельно измеряем выделение физических чанков.
        started = time.perf_counter()
        chunks = self.create_chunks(markdown)
        self.chunking_seconds = time.perf_counter() - started
        if not chunks:
            raise ValueError("Не удалось получить ни одного чанка из Markdown.")

        # Строим векторы и проверяем их соответствие физическим чанкам.
        started = time.perf_counter()
        embeddings = self.create_embeddings(markdown, chunks)
        self.embedding_seconds = time.perf_counter() - started
        hidden_size = int(self.embedder.model.config.hidden_size)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.shape != (len(chunks), hidden_size):
            raise ValueError(
                "Число или размерность эмбеддингов не соответствуют чанкам."
            )
        return chunks, embeddings


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
        chunks.append(IndexingStrategy.make_chunk(text, group, index, experiment))
    return chunks


class MarkdownHeaderSplitter(IndexingStrategy):
    """
    Строит прежние чанки по Markdown-заголовкам и токеновым ограничениям.

    Attributes:
        Наследует атрибуты IndexingStrategy.

    Fallbacks:
        Пустой Markdown обрабатывается общим контрактом стратегии.
    """

    def create_chunks(self, markdown: str) -> list[Chunk]:
        """
        Делегирует разбиение совместимому фасаду create_chunks.

        Parameters:
            markdown (str): Нормализованный Markdown.

        Returns:
            list[Chunk]: Чанки прежнего алгоритма.

        Fallbacks:
            Пустой документ возвращает пустой список.
        """
        return create_chunks(
            parse_markdown_units(markdown),
            self.experiment,
            self.embedder,
        )

    def create_embeddings(
        self,
        markdown: str,
        chunks: list[Chunk],
    ) -> object:
        """
        Строит прежние mean-pooled эмбеддинги чанков.

        Parameters:
            markdown (str): Нормализованный Markdown.
            chunks (list[Chunk]): Физические чанки.

        Returns:
            object: Матрица нормализованных эмбеддингов.

        Fallbacks:
            Поведение DenseEmbedder.embed сохраняется без изменений.
        """
        return self.embedder.embed(
            [chunk.text for chunk in chunks],
            batch_size=self.batch_size,
        )


class LongLateChunkingSplitter(IndexingStrategy):
    """
    Создаёт один физический чанк и один late-pooled вектор на рецепт.

    Attributes:
        Наследует атрибуты IndexingStrategy.

    Fallbacks:
        Документ без заголовков второго уровня отклоняется.
    """

    def create_chunks(self, markdown: str) -> list[Chunk]:
        """
        Группирует TextUnit от одного заголовка второго уровня до следующего.

        Parameters:
            markdown (str): Нормализованный Markdown.

        Returns:
            list[Chunk]: По одному полному физическому чанку на рецепт.

        Fallbacks:
            Служебный текст до первого рецепта пропускается.
        """
        # Группируем содержимое по паре «категория — рецепт».
        recipes: list[list[TextUnit]] = []
        current: list[TextUnit] = []
        current_key: tuple[str, ...] = ()
        for unit in parse_markdown_units(markdown):
            key = unit.headings[:2] if len(unit.headings) >= 2 else ()
            if not key:
                continue
            if current and key != current_key:
                recipes.append(current)
                current = []
            current_key = key
            current.append(unit)
        if current:
            recipes.append(current)
        if not recipes:
            raise ValueError("Markdown не содержит ни одного рецепта с заголовком ##.")

        # Восстанавливаем полные тексты, сохраняя вложенные заголовки ###.
        chunks: list[Chunk] = []
        for index, recipe in enumerate(recipes):
            parts = [f"# {recipe[0].headings[0]}", f"## {recipe[0].headings[1]}"]
            nested: tuple[str, ...] = ()
            for unit in recipe:
                if unit.headings[2:] != nested:
                    nested = unit.headings[2:]
                    parts.extend(
                        f"{'#' * min(level, 3)} {heading}"
                        for level, heading in enumerate(nested, start=3)
                    )
                parts.append(unit.text)
            text = "\n\n".join(parts).strip()
            chunks.append(
                self.make_chunk(
                    text,
                    recipe,
                    index,
                    self.experiment,
                    recipe[0].headings[:2],
                )
            )
        return chunks

    def create_embeddings(
        self,
        markdown: str,
        chunks: list[Chunk],
    ) -> object:
        """
        Усредняет контекстные токены окон, затем токены каждого рецепта.

        Parameters:
            markdown (str): Нормализованный Markdown.
            chunks (list[Chunk]): Полные рецепты в порядке документа.

        Returns:
            object: Нормализованная NumPy-матрица float32.

        Fallbacks:
            Недоступные offsets и непокрытые токены приводят к ValueError.
        """
        # Проверяем фактический контекст токенизатора до запуска инференса.
        special_count = self.embedder.tokenizer.num_special_tokens_to_add(pair=False)
        if self.experiment.chunk_size + special_count > self.embedder.max_length:
            raise ValueError(
                "Макроокно вместе со специальными токенами превышает контекст SBERT."
            )

        # Кодируем объединённые рецепты ровно один раз и фиксируем их char-границы.
        normalized = "\n\n".join(chunk.text for chunk in chunks)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for chunk in chunks:
            spans.append((cursor, cursor + len(chunk.text)))
            cursor += len(chunk.text) + 2
        encoded = self.embedder.tokenizer(
            normalized,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        if "offset_mapping" not in encoded:
            raise ValueError("Токенизатор SBERT не возвращает границы токенов.")
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        recipe_tokens = [
            [
                token_index
                for token_index, (start, end) in enumerate(offsets)
                if start < span_end and end > span_start
            ]
            for span_start, span_end in spans
        ]
        if any(not positions for positions in recipe_tokens):
            raise ValueError("Один из рецептов не содержит токенов для pooling.")

        # Формируем перекрывающиеся макроокна, включая последнее неполное окно.
        step = self.experiment.chunk_size - self.experiment.overlap
        window_count = (
            max(len(token_ids) - self.experiment.chunk_size, 0) + step - 1
        ) // step + 1
        window_starts = [index * step for index in range(window_count)]
        windows = [
            token_ids[start : start + self.experiment.chunk_size]
            for start in window_starts
            if token_ids[start : start + self.experiment.chunk_size]
        ]
        self.macro_window_count = len(windows)

        # Суммируем все контекстные представления каждого исходного токена.
        hidden_size = int(self.embedder.model.config.hidden_size)
        token_sums = np.zeros((len(token_ids), hidden_size), dtype=np.float32)
        token_counts = np.zeros(len(token_ids), dtype=np.int32)
        for offset in range(0, len(windows), self.batch_size):
            batch = windows[offset : offset + self.batch_size]
            representations = self.embedder.embed_token_batches(batch)
            for batch_index, values in enumerate(representations):
                start = window_starts[offset + batch_index]
                end = start + len(values)
                token_sums[start:end] += values
                token_counts[start:end] += 1
        if np.any(token_counts == 0):
            raise ValueError("Не все токены документа покрыты макроокнами.")
        token_vectors = token_sums / token_counts[:, None]

        # Усредняем уникальные токены рецепта и выполняем L2-нормализацию.
        vectors: list[object] = []
        for positions in recipe_tokens:
            vector = token_vectors[positions].mean(axis=0)
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Получен нулевой вектор рецепта.")
            vectors.append((vector / norm).astype(np.float32))
        return np.vstack(vectors).astype(np.float32)


def create_indexing_strategy(
    experiment: ExperimentConfig,
    embedder: DenseEmbedder,
    batch_size: int,
) -> IndexingStrategy:
    """
    Выбирает стратегию индексирования в единственной точке пайплайна.

    Parameters:
        experiment (ExperimentConfig): Параметры эксперимента.
        embedder (DenseEmbedder): Загруженный SBERT-эмбеддер.
        batch_size (int): Размер GPU-пакета.

    Returns:
        IndexingStrategy: Настроенная стратегия.

    Fallbacks:
        Неизвестное значение splitter отклоняется конфигурационным слоем.
    """
    strategies = {
        "markdown_header": MarkdownHeaderSplitter,
        "long_late_chunking": LongLateChunkingSplitter,
    }
    return strategies[experiment.splitter](experiment, embedder, batch_size)


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
    markdown = markdown_bytes.decode("utf-8")
    strategy = create_indexing_strategy(
        experiment,
        embedder,
        batch_size,
    )
    chunks, embeddings = strategy.build(markdown)

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
        "splitter": experiment.splitter,
        "chunk_count": len(chunks),
        "macro_window_count": strategy.macro_window_count,
        "chunking_seconds": strategy.chunking_seconds,
        "embedding_seconds": strategy.embedding_seconds,
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
