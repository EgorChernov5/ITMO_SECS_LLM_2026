"""Проверяет чанкинг и формулы retrieval-метрик."""

from source.scripts.build_index import create_chunks, parse_markdown_units
from source.scripts.evaluate_retrieval import calculate_metrics
from source.structures import ExperimentConfig


class FakeEmbedder:
    """
    Имитирует токенизатор эмбеддера пробельным разбиением.

    Attributes:
        Не предусмотрены.

    Fallbacks:
        Не предусмотрены.
    """

    def token_count(self, text: str) -> int:
        """
        Считает слова как токены.

        Parameters:
            text (str): Исходный текст.

        Returns:
            int: Число слов.

        Fallbacks:
            Пустой текст возвращает ноль.
        """
        return len(text.split())

    def encode_tokens(self, text: str) -> list[int]:
        """
        Кодирует слова последовательными индексами.

        Parameters:
            text (str): Исходный текст.

        Returns:
            list[int]: Тестовые токены.

        Fallbacks:
            Не предусмотрены.
        """
        return list(range(len(text.split())))

    def decode_tokens(self, tokens: list[int]) -> str:
        """
        Декодирует тестовые токены.

        Parameters:
            tokens (list[int]): Тестовые токены.

        Returns:
            str: Строка заданной длины.

        Fallbacks:
            Пустой список возвращает пустую строку.
        """
        return " ".join(f"токен{token}" for token in tokens)


def test_page_and_heading_parsing() -> None:
    """
    Проверяет перенос страниц и заголовков в текстовые единицы.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    markdown = "---\ntitle: test\n---\n<!-- page: 2 -->\n# Суп\nТекст"
    units = parse_markdown_units(markdown)
    assert units[0].page == 2
    assert units[0].headings == ("Суп",)


def test_chunk_ids_are_stable() -> None:
    """
    Проверяет стабильность идентификаторов чанков.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    units = parse_markdown_units(
        "<!-- page: 1 -->\n# Блюдо\nОдин два три четыре пять"
    )
    experiment = ExperimentConfig("test", 100, 10, 16, 100, 64)
    first = create_chunks(units, experiment, FakeEmbedder())
    second = create_chunks(units, experiment, FakeEmbedder())
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_retrieval_metrics() -> None:
    """
    Проверяет Precision, Recall и MRR по диапазонам страниц.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    results = [
        {"payload": {"page_start": 8, "page_end": 8}},
        {"payload": {"page_start": 2, "page_end": 3}},
        {"payload": {"page_start": 9, "page_end": 9}},
    ]
    metrics = calculate_metrics(results, (2, 3), 5)
    assert metrics["precision"] == 0.2
    assert metrics["recall"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5
