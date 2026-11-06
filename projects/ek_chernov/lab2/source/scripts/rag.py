"""Отвечает на вопросы по рецептам через Qdrant и Ollama."""

import argparse
import json
import os
from pathlib import Path

import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from source.configuration import (
    get_experiment_config,
    get_experiment_retrieval_config,
    get_ollama_context_window,
    load_pipeline_config,
)
from source.constants import (
    CONFIG_PATH,
    DEFAULT_CONFIG_NAME,
    GENERATION_RESPONSE_TOKEN_RESERVE,
    INVALID_EMPTY_QUESTION_RESPONSE,
    INVALID_LONG_QUESTION_RESPONSE,
    PROMPT_CONTEXT_OVERFLOW_MESSAGE,
    PROMPT_TOKEN_SAFETY_MARGIN,
    QWEN_TOKENIZER_NAME,
    RESULTS_DIR,
)
from source.embedding import DenseEmbedder
from source.retrieval import create_retriever
from source.vector_store import (
    collection_name,
    create_qdrant_client,
)


# Храним инструкцию отдельно для расчёта её токенов без дублирования текста.
RAG_INSTRUCTION = (
    "Ты отвечаешь на русском языке только по приведённому контексту. "
    "Не добавляй факты из общих знаний. Каждое содержательное утверждение "
    "сопровождай ссылкой вида [Рецепты.pdf, стр. N]. Если контекста "
    "недостаточно, прямо сообщи об этом."
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
    parser.add_argument("--top-k", type=int)
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
    best_path = RESULTS_DIR / "best_experiment.json"
    if not best_path.is_file():
        return DEFAULT_CONFIG_NAME
    try:
        data = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CONFIG_NAME
    if not isinstance(data, dict):
        return DEFAULT_CONFIG_NAME
    config = data.get("config")
    return config if isinstance(config, str) and config else DEFAULT_CONFIG_NAME


def build_prompt(
    question: str,
    results: list[dict[str, object]],
    tokenizer: PreTrainedTokenizerBase | None = None,
    context_window: int | None = None,
) -> str:
    """
    Собирает ограниченный найденными фрагментами промпт.

    Parameters:
        question (str): Вопрос пользователя.
        results (list[dict[str, object]]): Результаты Qdrant.
        tokenizer (PreTrainedTokenizerBase | None): Токенизатор Qwen для проверки
            итогового chat-промпта. По умолчанию: None.
        context_window (int | None): Размер контекстного окна Qwen в токенах.
            По умолчанию: None.

    Returns:
        str: Промпт для локальной LLM.

    Fallbacks:
        При пустом поиске контекст содержит явное сообщение об отсутствии данных.
        Не поместившиеся фрагменты удаляются с конца списка релевантности.
    """
    if (tokenizer is None) != (context_window is None):
        raise ValueError(
            "Токенизатор и context_window должны быть переданы одновременно."
        )
    prompt_token_limit = None
    if context_window is not None:
        prompt_token_limit = (
            context_window
            - GENERATION_RESPONSE_TOKEN_RESERVE
            - PROMPT_TOKEN_SAFETY_MARGIN
        )
        if prompt_token_limit <= 0:
            raise ValueError(
                "Контекстное окно не оставляет места для промпта после резервов."
            )

    selected_results = results.copy()
    while True:
        contexts: list[str] = []
        for number, result in enumerate(selected_results, start=1):
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
        if contexts:
            context = "\n\n".join(contexts)
        elif results:
            context = PROMPT_CONTEXT_OVERFLOW_MESSAGE
        else:
            context = "Релевантные фрагменты не найдены."
        prompt = f"{RAG_INSTRUCTION}\n\nКонтекст:\n{context}\n\nВопрос: {question}"
        if tokenizer is None or prompt_token_limit is None:
            return prompt
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(token_ids) <= prompt_token_limit:
            return prompt
        if selected_results:
            selected_results.pop()
            continue
        raise ValueError(
            "Инструкция и вопрос не помещаются в контекстное окно Qwen."
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
        "options": {
            "temperature": float(ollama.get("temperature", 0.2)),
            "num_ctx": get_ollama_context_window(settings),
            "num_predict": GENERATION_RESPONSE_TOKEN_RESERVE,
        },
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
        print(INVALID_EMPTY_QUESTION_RESPONSE)
        return
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("top-k должен быть положительным.")

    settings = load_pipeline_config(args.settings)
    context_window = get_ollama_context_window(settings)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_NAME)
    config_name = choose_config(args.config)
    experiment = get_experiment_config(settings, config_name)
    retrieval = get_experiment_retrieval_config(settings, experiment)
    embedder = DenseEmbedder(retrieval.bi_encoder.model)
    token_count = embedder.token_count(question)
    if token_count > embedder.max_length:
        print(
            INVALID_LONG_QUESTION_RESPONSE.format(
                token_count=token_count,
                max_tokens=embedder.max_length,
            )
        )
        return
    query_vector = embedder.embed([question], batch_size=1)[0].tolist()
    client = create_qdrant_client(settings)
    retriever = create_retriever(
        client,
        collection_name(settings, config_name),
        args.ef_search or experiment.ef_search,
        retrieval,
    )
    results = retriever.retrieve(question, query_vector, args.top_k)
    answer = ask_ollama(
        settings,
        build_prompt(question, results, qwen_tokenizer, context_window),
    )
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
