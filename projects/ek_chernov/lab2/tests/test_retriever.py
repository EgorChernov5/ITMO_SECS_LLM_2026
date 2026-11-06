"""Проверяет единый интерфейс Bi-Encoder и Cross-Encoder retrieval."""

from source.retrieval import (
    BiEncoderRetriever,
    CrossEncoderRetriever,
    Retriever,
    create_retriever,
)
from source.structures import RetrievalConfig, RetrievalModelConfig


class FakeCrossEncoder:
    """
    Возвращает заранее заданные оценки пар.

    Attributes:
        scores (list[float]): Оценки в порядке входных пар.
        pairs (list[tuple[str, str]]): Последние полученные пары.
        batch_size (int): Последний размер пакета.

    Fallbacks:
        Не предусмотрены.
    """

    def __init__(self, scores: list[float]) -> None:
        """
        Сохраняет тестовые оценки.

        Parameters:
            scores (list[float]): Возвращаемые оценки.

        Returns:
            None: Создаёт fake-модель.

        Fallbacks:
            Не предусмотрены.
        """
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []
        self.batch_size = 0

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int,
    ) -> list[float]:
        """
        Возвращает оценки и фиксирует аргументы.

        Parameters:
            pairs (list[tuple[str, str]]): Пары вопрос — документ.
            batch_size (int): Размер пакета.

        Returns:
            list[float]: Тестовые оценки.

        Fallbacks:
            Не предусмотрены.
        """
        self.pairs = pairs
        self.batch_size = batch_size
        return self.scores


def test_bi_retriever_preserves_score(monkeypatch: object) -> None:
    """
    Проверяет limit, CLI override и сохранение bi_score.

    Parameters:
        monkeypatch (object): Fixture подмены pytest.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    calls: list[int] = []

    # Подменяем только низкоуровневый Qdrant-поиск.
    def fake_search(
        client: object,
        name: str,
        vector: list[float],
        limit: int,
        ef_search: int,
    ) -> list[dict[str, object]]:
        calls.append(limit)
        return [
            {"id": str(index), "payload": {"text": str(index)}, "score": 1.0}
            for index in range(limit)
        ]

    monkeypatch.setattr("source.retrieval.search_chunks", fake_search)
    retriever = BiEncoderRetriever(object(), "recipes", 64, 5)
    default_results = retriever.retrieve("вопрос", [0.1])
    results = retriever.retrieve("вопрос", [0.1], 3)
    assert calls == [5, 3]
    assert len(default_results) == 5
    assert len(results) == 3
    assert all(result["score"] == result["bi_score"] for result in results)


def test_cross_retriever_reranks_stably(monkeypatch: object) -> None:
    """
    Проверяет candidate_top_k, оба score и стабильный порядок ties.

    Parameters:
        monkeypatch (object): Fixture подмены pytest.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    calls: list[int] = []

    # Возвращаем кандидатов в детерминированном dense-порядке.
    def fake_search(
        client: object,
        name: str,
        vector: list[float],
        limit: int,
        ef_search: int,
    ) -> list[dict[str, object]]:
        calls.append(limit)
        return [
            {
                "id": str(index),
                "payload": {"text": f"текст {index}"},
                "score": 1.0 - index / 10,
            }
            for index in range(limit)
        ]

    monkeypatch.setattr("source.retrieval.search_chunks", fake_search)
    retriever = CrossEncoderRetriever.__new__(CrossEncoderRetriever)
    Retriever.__init__(retriever, object(), "recipes", 64, 2)
    retriever.candidate_top_k = 4
    retriever.batch_size = 8
    retriever.cross_encoder = FakeCrossEncoder([0.2, 0.9, 0.9, 0.1])
    results = retriever.retrieve("вопрос", [0.1])
    assert calls == [4]
    assert [result["id"] for result in results] == ["1", "2"]
    assert results[0]["bi_score"] == 0.9
    assert results[0]["rerank_score"] == results[0]["score"] == 0.9
    assert retriever.cross_encoder.batch_size == 8


def test_factory_selects_bi_retriever() -> None:
    """
    Проверяет выбор BiEncoderRetriever фабрикой.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждением pytest.

    Fallbacks:
        Не предусмотрены.
    """
    bi_model = RetrievalModelConfig("sbert", "bi_encoder", "sbert", 8)
    config = RetrievalConfig("bi", "bi", bi_model, None, None, 5)
    assert isinstance(create_retriever(object(), "recipes", 64, config), BiEncoderRetriever)


def test_factory_selects_cross_retriever(monkeypatch: object) -> None:
    """
    Проверяет выбор CrossEncoderRetriever фабрикой без загрузки модели.

    Parameters:
        monkeypatch (object): Fixture подмены pytest.

    Returns:
        None: Проверка завершается утверждением pytest.

    Fallbacks:
        Реальная модель не загружается.
    """
    created: list[tuple[object, ...]] = []

    # Подменяем конструктор, чтобы проверить только выбор фабрики.
    def fake_cross(*args: object) -> object:
        created.append(args)
        return object()

    monkeypatch.setattr("source.retrieval.CrossEncoderRetriever", fake_cross)
    bi_model = RetrievalModelConfig("sbert", "bi_encoder", "sbert", 8)
    cross_model = RetrievalModelConfig(
        "reranker",
        "cross_encoder",
        "reranker",
        16,
    )
    config = RetrievalConfig(
        "bi_n_cross",
        "bi_n_cross",
        bi_model,
        cross_model,
        20,
        5,
    )
    result = create_retriever(object(), "recipes", 64, config)
    assert result is not None
    assert created[0][4:] == (20, "reranker", 16)
