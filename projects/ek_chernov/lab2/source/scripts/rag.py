"""Отвечает на вопросы по рецептам через Qdrant и Ollama."""

import argparse
import json
import os
from pathlib import Path

import requests

from source.configuration import get_experiment_config, load_pipeline_config
from source.constants import (
    CONFIG_PATH,
    DEFAULT_CONFIG_NAME,
    MODEL_NAME,
    RESULTS_DIR,
)
from source.embedding import DenseEmbedder
from source.vector_store import (
    collection_name,
    create_qdrant_client,
    search_chunks,
)


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры RAG-запроса.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы поискового запроса.

    Fallbacks:
        Конфигурация выбирается по результатам оценки или baseline.
    """
    parser = argparse.ArgumentParser(description="RAG по документу с рецептами.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--config")
    parser.add_argument("--settings", type=Path, default=CONFIG_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ef-search", type=int)
    return parser.parse_args()


def choose_config(explicit: str | None) -> str:
    """
    Выбирает экспериментальную конфигурацию.

    Parameters:
        explicit (str | None): Явно переданное название.

    Returns:
        str: Название выбранной конфигурации.

    Fallbacks:
        Использует baseline при отсутствии результатов оценки.
    """
    if explicit:
        return explicit
    best_path = RESULTS_DIR / "best_config.json"
    if not best_path.is_file():
        return DEFAULT_CONFIG_NAME
    data = json.loads(best_path.read_text(encoding="utf-8"))
    return str(data.get("config", DEFAULT_CONFIG_NAME))


def build_prompt(
    question: str,
    results: list[dict[str, object]],
) -> str:
    """
    Собирает ограниченный найденными фрагментами промпт.

    Parameters:
        question (str): Вопрос пользователя.
        results (list[dict[str, object]]): Результаты Qdrant.

    Returns:
        str: Промпт для локальной LLM.

    Fallbacks:
        При пустом поиске контекст содержит явное сообщение об отсутствии данных.
    """
    contexts: list[str] = []
    for number, result in enumerate(results, start=1):
        payload = result["payload"]
        page_start = int(payload["page_start"])
        page_end = int(payload["page_end"])
        page_label = (
            str(page_start)
            if page_start == page_end
            else f"{page_start}–{page_end}"
        )
        contexts.append(
            f"[Фрагмент {number}; Рецепты.pdf, стр. {page_label}]\n"
            f"{payload['text']}"
        )
    context = "\n\n".join(contexts) or "Релевантные фрагменты не найдены."
    return (
        "Ты отвечаешь на русском языке только по приведённому контексту. "
        "Не добавляй факты из общих знаний. Каждое содержательное утверждение "
        "сопровождай ссылкой вида [Рецепты.pdf, стр. N]. Если контекста "
        "недостаточно, прямо сообщи об этом.\n\n"
        f"Контекст:\n{context}\n\nВопрос: {question}"
    )


def ask_ollama(
    settings: dict[str, object],
    prompt: str,
) -> str:
    """
    Отправляет промпт локальной модели Ollama.

    Parameters:
        settings (dict[str, object]): Конфигурация пайплайна.
        prompt (str): Подготовленный RAG-промпт.

    Returns:
        str: Ответ модели.

    Fallbacks:
        Сетевые и HTTP-ошибки преобразуются в RuntimeError.
    """
    ollama = settings.get("ollama", {})
    if not isinstance(ollama, dict):
        raise ValueError("Секция ollama должна быть JSON-объектом.")
    base_url = os.getenv(
        "OLLAMA_URL",
        str(ollama.get("url", "http://localhost:11434")),
    ).rstrip("/")
    payload = {
        "model": str(ollama.get("model", "qwen2.5:3b-instruct")),
        "stream": False,
        "options": {"temperature": float(ollama.get("temperature", 0.2))},
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["message"]["content"]).strip()
    except (requests.RequestException, KeyError, ValueError) as error:
        raise RuntimeError(
            "Не удалось получить ответ Ollama. Проверьте сервис и модель "
            "qwen2.5:3b-instruct."
        ) from error


def main() -> None:
    """
    Выполняет поиск и генерацию ответа с цитатами.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Печатает ответ и найденные источники.

    Fallbacks:
        До появления результатов оценки используется baseline.
    """
    args = parse_args()
    question = args.question.strip()
    if not question:
        raise ValueError("Вопрос не должен быть пустым.")
    if args.top_k <= 0:
        raise ValueError("top-k должен быть положительным.")

    settings = load_pipeline_config(args.settings)
    config_name = choose_config(args.config)
    experiment = get_experiment_config(settings, config_name)
    embedding = settings.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("Секция embedding должна быть JSON-объектом.")
    embedder = DenseEmbedder(str(embedding.get("model", MODEL_NAME)))
    query_vector = embedder.embed([question], batch_size=1)[0].tolist()
    client = create_qdrant_client(settings)
    results = search_chunks(
        client,
        collection_name(settings, config_name),
        query_vector,
        args.top_k,
        args.ef_search or experiment.ef_search,
    )
    answer = ask_ollama(settings, build_prompt(question, results))
    print(answer)
    print("\nИсточники:")
    for result in results:
        payload = result["payload"]
        snippet = " ".join(str(payload["text"]).split())[:220]
        print(
            f"- Рецепты.pdf, стр. {payload['page_start']}"
            f"–{payload['page_end']}; score={result['score']:.4f}; "
            f"{snippet}"
        )


if __name__ == "__main__":
    main()
