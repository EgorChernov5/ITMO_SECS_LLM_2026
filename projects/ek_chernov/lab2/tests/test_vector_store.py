"""Проверяет ожидание готовности удалённого Qdrant."""

from types import SimpleNamespace

from source.vector_store import wait_for_collection_index


class FakeQdrantClient:
    """
    Имитирует полностью проиндексированную коллекцию.

    Attributes:
        calls (int): Число запросов состояния коллекции.

    Fallbacks:
        Не предусмотрены.
    """

    def __init__(self) -> None:
        """
        Создаёт тестовый клиент.

        Parameters:
            Не предусмотрены.

        Returns:
            None: Инициализирует счётчик вызовов.

        Fallbacks:
            Не предусмотрены.
        """
        self.calls = 0

    def get_collection(self, name: str) -> object:
        """
        Возвращает состояние готового HNSW.

        Parameters:
            name (str): Название коллекции.

        Returns:
            object: Состояние коллекции.

        Fallbacks:
            Не предусмотрены.
        """
        assert name == "recipes_test"
        self.calls += 1
        return SimpleNamespace(
            status=SimpleNamespace(value="green"),
            indexed_vectors_count=15,
        )


def test_ready_collection_does_not_wait() -> None:
    """
    Проверяет немедленное завершение для готовой коллекции.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждением pytest.

    Fallbacks:
        Не предусмотрены.
    """
    client = FakeQdrantClient()
    wait_for_collection_index(client, "recipes_test", 15, 1)
    assert client.calls == 1
