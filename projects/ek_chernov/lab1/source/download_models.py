"""Загружает требуемые модели через HTTP API Ollama."""

import http.client
import json
import urllib.error
import urllib.request

from constants import MODELS, OLLAMA_URL


def api_request(
    path: str,
    data: dict[str, object] | None = None,
) -> http.client.HTTPResponse:
    """
    Выполняет запрос к HTTP API Ollama.

    Parameters:
        path (str): Путь метода API.
        data (dict[str, object] | None): Тело запроса. По умолчанию: None.

    Returns:
        http.client.HTTPResponse: Поток ответа сервера.

    Fallbacks:
        При сетевой ошибке исключение передаётся вызывающему коду.
    """
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=body,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    return urllib.request.urlopen(request, timeout=120)


def installed_models() -> set[str]:
    """
    Возвращает идентификаторы моделей, доступных в Ollama.

    Returns:
        set[str]: Множество идентификаторов установленных моделей.

    Fallbacks:
        При ошибке API исключение передаётся вызывающему коду.
    """
    with api_request("/api/tags") as response:
        data = json.load(response)
    return {model["name"] for model in data.get("models", [])}


def download_model(model: str) -> None:
    """
    Загружает модель и выводит прогресс операции.

    Parameters:
        model (str): Идентификатор загружаемой модели.

    Returns:
        None: Функция не возвращает значение.

    Fallbacks:
        Ошибка из потока Ollama преобразуется в RuntimeError.
    """
    print(f"\nЗагрузка модели: {model}")
    with api_request(
        "/api/pull",
        {"model": model, "stream": True},
    ) as response:
        for line in response:
            if not line.strip():
                continue

            data = json.loads(line)
            status = data.get("status", "")
            if "error" in data:
                raise RuntimeError(data["error"])

            total = data.get("total", 0)
            completed = data.get("completed", 0)
            if total:
                percent = completed / total * 100
                print(f"\r{status}: {percent:.1f}%", end="", flush=True)
            else:
                print(f"\n{status}")

    print(f"\nМодель {model} загружена")


def main() -> None:
    """
    Загружает отсутствующие модели и проверяет итоговый список.

    Returns:
        None: Функция не возвращает значение.

    Fallbacks:
        Ошибка подключения завершает сценарий, а ошибка отдельной загрузки
        выводится без прерывания обработки остальных моделей.
    """
    try:
        installed = installed_models()
    except urllib.error.URLError as error:
        print(f"Не удалось подключиться к Ollama: {error}")
        return

    print("Уже установленные модели:")
    for model in sorted(installed):
        print(f"  - {model}")

    for model in MODELS:
        if model in installed:
            print(f"\n{model} уже установлена. Пропускаем.")
            continue

        try:
            download_model(model)
        except Exception as error:
            print(f"\nОшибка загрузки {model}: {error}")

    print("\nПроверка итогового списка моделей:")
    final_models = installed_models()
    for model in MODELS:
        status = "ЗАГРУЖЕНА" if model in final_models else "НЕ ЗАГРУЖЕНА"
        print(f"  {model}: {status}")


if __name__ == "__main__":
    main()
