"""Содержит структуры результатов эксперимента."""

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RequestMetrics:
    """
    Хранит метрики и ответ одного запроса к модели.

    Attributes:
        answer (str): Текст ответа модели.
        time_seconds (float): Полное время HTTP-запроса в секундах.
        prompt_tokens (int | None): Количество входных токенов.
        completion_tokens (int | None): Количество выходных токенов.
        tokens_per_second (float | None): Скорость генерации по полному времени.
        finish_reason (str | None): Причина завершения генерации.

    Fallbacks:
        Недоступные метрики использования сохраняются как None.
    """

    answer: str
    time_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    finish_reason: str | None


@dataclass(slots=True)
class ExperimentRecord:
    """
    Хранит результат одного прогона модели.

    Attributes:
        model (str): Идентификатор модели.
        task (str): Идентификатор задания.
        mode (str): Название режима генерации.
        repeat (int): Номер повтора.
        prompt (str): Отправленный модели запрос.
        parameters (dict[str, int | float]): Параметры генерации.
        answer (str | None): Ответ модели или None при ошибке.
        time_seconds (float | None): Время запроса в секундах.
        prompt_tokens (int | None): Количество входных токенов.
        completion_tokens (int | None): Количество выходных токенов.
        tokens_per_second (float | None): Скорость генерации.
        finish_reason (str | None): Причина завершения генерации.
        error (str | None): Текст ошибки или None при успехе.

    Fallbacks:
        При ошибке запроса метрики и ответ сохраняются как None.
    """

    model: str
    task: str
    mode: str
    repeat: int
    prompt: str
    parameters: dict[str, int | float]
    answer: str | None
    time_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    finish_reason: str | None
    error: str | None

    def to_dict(self: "ExperimentRecord") -> dict[str, object]:
        """
        Преобразует результат прогона в словарь для сериализации.

        Returns:
            dict[str, object]: Словарь со всеми полями результата.

        Fallbacks:
            Не предусмотрены.
        """
        return asdict(self)
