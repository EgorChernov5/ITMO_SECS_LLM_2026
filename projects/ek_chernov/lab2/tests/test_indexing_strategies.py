"""Проверяет границы рецептов и агрегацию Long Late Chunking."""

import re
from types import SimpleNamespace

import numpy as np

from source.scripts.build_index import LongLateChunkingSplitter
from source.structures import ExperimentConfig


class FakeTokenizer:
    """
    Имитирует fast-токенизатор с offset mapping.

    Attributes:
        Не предусмотрены.

    Fallbacks:
        Любая непробельная последовательность считается одним токеном.
    """

    def __call__(self, text: str, **options: object) -> dict[str, object]:
        """
        Кодирует текст и возвращает символьные границы токенов.

        Parameters:
            text (str): Исходный текст.
            options (object): Совместимые параметры токенизатора.

        Returns:
            dict[str, object]: Идентификаторы и offset mapping.

        Fallbacks:
            Пустой текст возвращает пустые последовательности.
        """
        # Нумеруем токены в порядке документа для проверяемого pooling.
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(1, len(matches) + 1)),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Возвращает число служебных токенов тестовой модели.

        Parameters:
            pair (bool): Признак пары текстов. По умолчанию: False.

        Returns:
            int: Два служебных токена.

        Fallbacks:
            Значение не зависит от pair.
        """
        return 2


class FakeLongEmbedder:
    """
    Возвращает детерминированные контекстные token-level векторы.

    Attributes:
        tokenizer (FakeTokenizer): Тестовый токенизатор.
        max_length (int): Контекст с учётом служебных токенов.
        model (SimpleNamespace): Конфигурация с hidden_size.

    Fallbacks:
        Не выполняет GPU-инференс.
    """

    def __init__(self) -> None:
        """
        Создаёт тестовый эмбеддер размерности два.

        Parameters:
            Не предусмотрены.

        Returns:
            None: Инициализирует зависимости теста.

        Fallbacks:
            Не предусмотрены.
        """
        self.tokenizer = FakeTokenizer()
        self.max_length = 7
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))

    def embed_token_batches(
        self,
        token_batches: list[list[int]],
    ) -> list[object]:
        """
        Строит вектор из ID токена и его позиции внутри окна.

        Parameters:
            token_batches (list[list[int]]): Макроокна без special tokens.

        Returns:
            list[object]: Token-level матрицы float32.

        Fallbacks:
            Пустой пакет возвращает пустой список.
        """
        # Позиционная координата различается для токенов overlap в соседних окнах.
        return [
            np.asarray(
                [[token, position + 1] for position, token in enumerate(window)],
                dtype=np.float32,
            )
            for window in token_batches
        ]


def experiment() -> ExperimentConfig:
    """
    Возвращает конфигурацию Long Late Chunking для тестов.

    Parameters:
        Не предусмотрены.

    Returns:
        ExperimentConfig: Конфигурация окна 5 с overlap 2.

    Fallbacks:
        Не предусмотрены.
    """
    return ExperimentConfig(
        "late", "long_late_chunking", 5, 2, 16, 100, 64, "bi"
    )


def test_recipe_boundaries_preserve_nested_headings() -> None:
    """
    Проверяет один чанк на ##, сохранение ### и смену категории.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Служебный текст до первого ## не становится рецептом.
    """
    # Документ содержит служебный текст, два рецепта и две категории.
    markdown = """Служебный текст
# Супы
Описание категории
## Борщ
<!-- page: 2 -->
Ингредиенты
### Приготовление
Варить
## Щи
<!-- page: 3 -->
Тушить
# Выпечка
## Пирог
<!-- page: 4 -->
Печь
"""
    strategy = LongLateChunkingSplitter(experiment(), FakeLongEmbedder(), 2)
    chunks = strategy.create_chunks(markdown)
    assert [chunk.headings for chunk in chunks] == [
        ("Супы", "Борщ"),
        ("Супы", "Щи"),
        ("Выпечка", "Пирог"),
    ]
    assert "### Приготовление" in chunks[0].text
    assert chunks[0].page_start == chunks[0].page_end == 2


def test_overlap_is_averaged_before_recipe_pooling() -> None:
    """
    Проверяет равный вес токенов независимо от числа покрывающих окон.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка сравнивает результат с независимым расчётом.

    Fallbacks:
        Контекстные значения задаются детерминированным фейком.
    """
    # Строим один рецепт, который пересекает несколько макроокон.
    markdown = "# Категория\n## Рецепт\nодин два три четыре пять шесть"
    embedder = FakeLongEmbedder()
    strategy = LongLateChunkingSplitter(experiment(), embedder, 2)
    chunks = strategy.create_chunks(markdown)
    vectors = strategy.create_embeddings(markdown, chunks)

    # Независимо усредняем контекстные дубли каждого исходного токена.
    encoded = embedder.tokenizer(chunks[0].text)
    token_ids = encoded["input_ids"]
    step = experiment().chunk_size - experiment().overlap
    window_count = (
        max(len(token_ids) - experiment().chunk_size, 0) + step - 1
    ) // step + 1
    sums = np.zeros((len(token_ids), 2), dtype=np.float32)
    counts = np.zeros(len(token_ids), dtype=np.int32)
    for start in [index * step for index in range(window_count)]:
        window = token_ids[start : start + experiment().chunk_size]
        values = embedder.embed_token_batches([window])[0]
        sums[start : start + len(window)] += values
        counts[start : start + len(window)] += 1
    expected = (sums / counts[:, None]).mean(axis=0)
    expected = expected / np.linalg.norm(expected)

    assert vectors.dtype == np.float32
    assert vectors.shape == (1, 2)
    assert np.allclose(vectors[0], expected)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert strategy.macro_window_count == window_count
