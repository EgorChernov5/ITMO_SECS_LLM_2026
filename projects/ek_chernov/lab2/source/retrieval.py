"""Содержит единый интерфейс dense retrieval и реранжирования."""

from source.structures import RetrievalConfig
from source.vector_store import search_chunks


class Retriever:
    """
    Задаёт общий интерфейс retrieval для Qdrant.

    Attributes:
        client (object): Клиент Qdrant.
        collection (str): Имя коллекции.
        ef_search (int): Параметр поиска HNSW.
        top_k (int): Итоговый размер выдачи.

    Fallbacks:
        Неположительный итоговый top_k отклоняется до запроса к Qdrant.
    """

    def __init__(
        self,
        client: object,
        collection: str,
        ef_search: int,
        top_k: int,
    ) -> None:
        """
        Сохраняет общие параметры retrieval.

        Parameters:
            client (object): Клиент Qdrant.
            collection (str): Имя коллекции.
            ef_search (int): Параметр поиска HNSW.
            top_k (int): Итоговый размер выдачи.

        Returns:
            None: Создаёт retriever.

        Fallbacks:
            Валидация top_k выполняется при retrieval.
        """
        self.client = client
        self.collection = collection
        self.ef_search = ef_search
        self.top_k = top_k

    def resolve_top_k(self, top_k: int | None) -> int:
        """
        Разрешает CLI override итогового размера выдачи.

        Parameters:
            top_k (int | None): Явное переопределение. По умолчанию: None.

        Returns:
            int: Проверенный итоговый размер выдачи.

        Fallbacks:
            При отсутствии override использует значение конфигурации.
        """
        resolved = self.top_k if top_k is None else top_k
        if resolved <= 0:
            raise ValueError("top_k должен быть положительным.")
        return resolved

    def retrieve(
        self,
        question: str,
        vector: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, object]]:
        """
        Возвращает ранжированные чанки для вопроса.

        Parameters:
            question (str): Исходный вопрос.
            vector (list[float]): Готовый dense-вектор вопроса.
            top_k (int | None): CLI override размера выдачи. По умолчанию: None.

        Returns:
            list[dict[str, object]]: Результаты с id, payload и score.

        Fallbacks:
            Реализация задаётся подклассом.
        """
        raise NotImplementedError


class BiEncoderRetriever(Retriever):
    """
    Выполняет dense-поиск ближайших чанков в Qdrant.

    Attributes:
        Наследует атрибуты Retriever.

    Fallbacks:
        Ошибки Qdrant передаются вызывающему коду.
    """

    def retrieve(
        self,
        question: str,
        vector: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, object]]:
        """
        Ищет итоговый top_k и сохраняет исходный bi_score.

        Parameters:
            question (str): Исходный вопрос.
            vector (list[float]): Готовый dense-вектор вопроса.
            top_k (int | None): CLI override размера выдачи. По умолчанию: None.

        Returns:
            list[dict[str, object]]: Результаты dense-поиска.

        Fallbacks:
            Текст вопроса не используется dense-стратегией.
        """
        limit = self.resolve_top_k(top_k)
        results = search_chunks(
            self.client,
            self.collection,
            vector,
            limit,
            self.ef_search,
        )
        for result in results:
            result["bi_score"] = result["score"]
        return results


class CrossEncoderRetriever(BiEncoderRetriever):
    """
    Переранжирует расширенный dense-список Cross-Encoder моделью.

    Attributes:
        candidate_top_k (int): Число dense-кандидатов.
        batch_size (int): Размер пакета Cross-Encoder.
        cross_encoder (object): Загруженная модель реранжирования.

    Fallbacks:
        Ошибка загрузки модели приводит к явному RuntimeError без fallback.
    """

    def __init__(
        self,
        client: object,
        collection: str,
        ef_search: int,
        top_k: int,
        candidate_top_k: int,
        model_name: str,
        batch_size: int,
    ) -> None:
        """
        Один раз загружает Cross-Encoder и сохраняет параметры поиска.

        Parameters:
            client (object): Клиент Qdrant.
            collection (str): Имя коллекции.
            ef_search (int): Параметр поиска HNSW.
            top_k (int): Итоговый размер выдачи.
            candidate_top_k (int): Размер dense-списка кандидатов.
            model_name (str): Идентификатор Cross-Encoder.
            batch_size (int): Размер пакета инференса.

        Returns:
            None: Создаёт готовый retriever.

        Fallbacks:
            Импорт и загрузка модели преобразуются в RuntimeError.
        """
        super().__init__(client, collection, ef_search, top_k)
        self.candidate_top_k = candidate_top_k
        self.batch_size = batch_size
        try:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(model_name)
        except Exception as error:
            raise RuntimeError(
                f"Не удалось загрузить Cross-Encoder {model_name!r}. "
                "Проверьте sentence-transformers, сеть и доступ к модели."
            ) from error

    def retrieve(
        self,
        question: str,
        vector: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, object]]:
        """
        Получает dense-кандидатов и сортирует их по rerank_score.

        Parameters:
            question (str): Исходный вопрос.
            vector (list[float]): Готовый dense-вектор вопроса.
            top_k (int | None): CLI override размера выдачи. По умолчанию: None.

        Returns:
            list[dict[str, object]]: Переранжированный итоговый список.

        Fallbacks:
            Стабильная сортировка сохраняет dense-порядок при равных score.
        """
        limit = self.resolve_top_k(top_k)
        candidates = super().retrieve(
            question,
            vector,
            max(self.candidate_top_k, limit),
        )
        # Не вызываем модель для пустого ответа Qdrant.
        if not candidates:
            return []
        pairs = [
            (question, str(result["payload"]["text"]))
            for result in candidates
        ]
        scores = self.cross_encoder.predict(pairs, batch_size=self.batch_size)
        # Не допускаем незаметной потери кандидатов при ошибочном ответе модели.
        if len(scores) != len(candidates):
            raise RuntimeError(
                "Cross-Encoder вернул число оценок, не совпадающее с кандидатами."
            )
        for result, score in zip(candidates, scores):
            result["rerank_score"] = float(score)
            result["score"] = float(score)
        candidates.sort(key=lambda result: result["rerank_score"], reverse=True)
        return candidates[:limit]


def create_retriever(
    client: object,
    collection: str,
    ef_search: int,
    config: RetrievalConfig,
) -> Retriever:
    """
    Создаёт retriever требуемой стратегии в единственной точке выбора.

    Parameters:
        client (object): Клиент Qdrant.
        collection (str): Имя коллекции.
        ef_search (int): Параметр поиска HNSW.
        config (RetrievalConfig): Разрешённая retrieval-конфигурация.

    Returns:
        Retriever: Dense или Cross-Encoder retriever.

    Fallbacks:
        Неизвестная стратегия приводит к ValueError без fallback.
    """
    if config.type == "bi":
        return BiEncoderRetriever(client, collection, ef_search, config.top_k)
    if (
        config.type == "bi_n_cross"
        and config.cross_encoder is not None
        and config.candidate_top_k is not None
    ):
        return CrossEncoderRetriever(
            client,
            collection,
            ef_search,
            config.top_k,
            config.candidate_top_k,
            config.cross_encoder.model,
            config.cross_encoder.batch_size,
        )
    raise ValueError(f"Неизвестный тип retrieval: {config.type!r}.")
