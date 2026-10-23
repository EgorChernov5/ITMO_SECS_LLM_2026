"""Оценивает retrieval для экспериментальных конфигураций."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

from source.configuration import get_experiment_config, load_pipeline_config
from source.constants import CONFIG_PATH, MODEL_NAME, QUESTIONS_PATH, RESULTS_DIR
from source.embedding import DenseEmbedder
from source.structures import EvaluationQuestion
from source.vector_store import (
    collection_name,
    create_qdrant_client,
    load_latest_build,
    qdrant_settings,
    search_chunks,
    wait_for_collection_index,
)


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры оценки retrieval.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы оценки.

    Fallbacks:
        По умолчанию оцениваются все три конфигурации.
    """
    parser = argparse.ArgumentParser(
        description="Оценка качества retrieval в Qdrant."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["small", "baseline", "large"],
    )
    parser.add_argument("--settings", type=Path, default=CONFIG_PATH)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--repetitions", type=int)
    return parser.parse_args()


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """
    Загружает ручную разметку оценочных вопросов.

    Parameters:
        path (Path): Путь к JSON с вопросами.

    Returns:
        list[EvaluationQuestion]: Проверенный набор из 10–20 вопросов.

    Fallbacks:
        При неполной разметке возбуждает ValueError.
    """
    try:
        raw_questions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Не удалось прочитать вопросы {path}: {error}") from error
    if not isinstance(raw_questions, list) or not 10 <= len(raw_questions) <= 20:
        raise ValueError("Разметка должна содержать от 10 до 20 вопросов.")

    questions: list[EvaluationQuestion] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise ValueError("Каждый вопрос должен быть JSON-объектом.")
        pages = tuple(sorted({int(page) for page in raw["relevant_pages"]}))
        if not pages or min(pages) < 1:
            raise ValueError("У каждого вопроса должна быть релевантная страница.")
        questions.append(
            EvaluationQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]).strip(),
                relevant_pages=pages,
            )
        )
    if len({question.id for question in questions}) != len(questions):
        raise ValueError("Идентификаторы вопросов должны быть уникальными.")
    return questions


def calculate_metrics(
    results: list[dict[str, object]],
    relevant_pages: tuple[int, ...],
    k: int,
) -> dict[str, float]:
    """
    Вычисляет Precision@k, Recall@k и MRR.

    Parameters:
        results (list[dict[str, object]]): Ранжированные результаты поиска.
        relevant_pages (tuple[int, ...]): Эталонные страницы.
        k (int): Глубина оценки.

    Returns:
        dict[str, float]: Значения precision, recall и reciprocal_rank.

    Fallbacks:
        При отсутствии результатов возвращает нулевые метрики.
    """
    expected = set(relevant_pages)
    retrieved_pages: set[int] = set()
    relevant_chunks = 0
    reciprocal_rank = 0.0
    for rank, result in enumerate(results[:k], start=1):
        payload = result.get("payload", {})
        if not isinstance(payload, dict):
            continue
        page_start = int(payload.get("page_start", 0))
        page_end = int(payload.get("page_end", page_start))
        matched = expected.intersection(range(page_start, page_end + 1))
        if matched:
            relevant_chunks += 1
            retrieved_pages.update(matched)
            if reciprocal_rank == 0:
                reciprocal_rank = 1 / rank
    return {
        "precision": relevant_chunks / k,
        "recall": len(retrieved_pages) / len(expected),
        "reciprocal_rank": reciprocal_rank,
    }


def evaluate_config(
    client: object,
    settings: dict[str, object],
    config_name: str,
    questions: list[EvaluationQuestion],
    vectors: object,
    warmup_runs: int,
    repetitions: int,
) -> dict[str, object]:
    """
    Оценивает одну прогретую коллекцию Qdrant несколькими повторами.

    Parameters:
        client (object): Клиент Qdrant.
        settings (dict[str, object]): Конфигурация пайплайна.
        config_name (str): Название эксперимента.
        questions (list[EvaluationQuestion]): Оценочные вопросы.
        vectors (object): Матрица эмбеддингов вопросов.
        warmup_runs (int): Число прогревочных запросов на вопрос.
        repetitions (int): Число измеряемых запросов на вопрос.

    Returns:
        dict[str, object]: Детальные и агрегированные метрики.

    Fallbacks:
        Оценка не начинается до полной готовности HNSW.
    """
    experiment = get_experiment_config(settings, config_name)
    _, manifest, _, _ = load_latest_build(config_name)
    qdrant = qdrant_settings(settings)
    wait_for_collection_index(
        client,
        collection_name(settings, config_name),
        int(manifest["chunk_count"]),
        float(qdrant.get("indexing_wait_timeout_seconds", 180)),
    )
    per_question: list[dict[str, object]] = []
    latencies_ms: list[float] = []

    for index, question in enumerate(questions):
        vector = vectors[index].tolist()
        name = collection_name(settings, config_name)
        for _ in range(warmup_runs):
            search_chunks(
                client,
                name,
                vector,
                limit=10,
                ef_search=experiment.ef_search,
            )

        results: list[dict[str, object]] = []
        question_latencies: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            current_results = search_chunks(
                client,
                name,
                vector,
                limit=10,
                ef_search=experiment.ef_search,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            question_latencies.append(elapsed_ms)
            latencies_ms.append(elapsed_ms)
            if not results:
                results = current_results

        at_5 = calculate_metrics(results, question.relevant_pages, 5)
        at_10 = calculate_metrics(results, question.relevant_pages, 10)
        per_question.append(
            {
                "id": question.id,
                "question": question.question,
                "relevant_pages": list(question.relevant_pages),
                "precision_at_5": at_5["precision"],
                "recall_at_5": at_5["recall"],
                "precision_at_10": at_10["precision"],
                "recall_at_10": at_10["recall"],
                "reciprocal_rank": at_10["reciprocal_rank"],
                "median_search_ms": median(question_latencies),
            }
        )

    count = len(per_question)
    aggregate = {
        key: sum(float(item[key]) for item in per_question) / count
        for key in (
            "precision_at_5",
            "recall_at_5",
            "precision_at_10",
            "recall_at_10",
            "reciprocal_rank",
        )
    }
    aggregate["average_search_ms"] = mean(latencies_ms)
    aggregate["median_search_ms"] = median(latencies_ms)
    return {
        "config": config_name,
        "parameters": {
            "chunk_size": experiment.chunk_size,
            "overlap": experiment.overlap,
            "hnsw_m": experiment.hnsw_m,
            "ef_construct": experiment.ef_construct,
            "ef_search": experiment.ef_search,
            "warmup_runs": warmup_runs,
            "repetitions": repetitions,
        },
        "costs": {
            "chunk_count": manifest["chunk_count"],
            "chunking_seconds": manifest["chunking_seconds"],
            "embedding_seconds": manifest["embedding_seconds"],
        },
        "aggregate": aggregate,
        "questions": per_question,
    }


def save_results(results: list[dict[str, object]]) -> str:
    """
    Сохраняет JSON, Markdown-отчёт и лучшую конфигурацию.

    Parameters:
        results (list[dict[str, object]]): Результаты всех экспериментов.

    Returns:
        str: Название лучшей конфигурации.

    Fallbacks:
        Не предусмотрены.
    """
    best = max(
        results,
        key=lambda result: (
            float(result["aggregate"]["reciprocal_rank"]),
            float(result["aggregate"]["recall_at_5"]),
            -float(result["aggregate"]["median_search_ms"]),
        ),
    )
    best_name = str(best["config"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (RESULTS_DIR / "best_config.json").write_text(
        json.dumps({"config": best_name}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rows = [
        "# Сравнение конфигураций retrieval",
        "",
        "| Конфигурация | P@5 | R@5 | P@10 | R@10 | MRR | Медиана, мс | Чанки |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        aggregate = result["aggregate"]
        costs = result["costs"]
        rows.append(
            f"| {result['config']} | "
            f"{aggregate['precision_at_5']:.4f} | "
            f"{aggregate['recall_at_5']:.4f} | "
            f"{aggregate['precision_at_10']:.4f} | "
            f"{aggregate['recall_at_10']:.4f} | "
            f"{aggregate['reciprocal_rank']:.4f} | "
            f"{aggregate['median_search_ms']:.2f} | "
            f"{costs['chunk_count']} |"
        )
    rows.extend(
        [
            "",
            "## Вывод",
            "",
            f"Лучшая конфигурация: **{best_name}**. "
            "Выбор выполнен по MRR, затем Recall@5 и медиане времени поиска.",
            "",
            "Время измерено на клиенте и включает задержку SSH-туннеля.",
            "",
        ]
    )
    (RESULTS_DIR / "retrieval_report.md").write_text(
        "\n".join(rows),
        encoding="utf-8",
        newline="\n",
    )
    return best_name


def main() -> None:
    """
    Запускает оценку всех указанных конфигураций.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Сохраняет и печатает результаты эксперимента.

    Fallbacks:
        Число прогревов и повторов берётся из конфигурации.
    """
    args = parse_args()
    settings = load_pipeline_config(args.settings)
    questions = load_questions(args.questions)
    evaluation = settings.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("Секция evaluation должна быть JSON-объектом.")
    warmup_runs = (
        args.warmup_runs
        if args.warmup_runs is not None
        else int(evaluation.get("warmup_runs", 1))
    )
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(evaluation.get("repetitions", 5))
    )
    if warmup_runs < 0:
        raise ValueError("Число прогревов не может быть отрицательным.")
    if repetitions <= 0:
        raise ValueError("Число повторов должно быть положительным.")

    embedding = settings.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("Секция embedding должна быть JSON-объектом.")
    embedder = DenseEmbedder(str(embedding.get("model", MODEL_NAME)))
    vectors = embedder.embed(
        [question.question for question in questions],
        batch_size=int(embedding.get("batch_size", 8)),
    )
    client = create_qdrant_client(settings)
    results = [
        evaluate_config(
            client,
            settings,
            config_name,
            questions,
            vectors,
            warmup_runs,
            repetitions,
        )
        for config_name in args.configs
    ]
    best_name = save_results(results)
    print(f"Оценка завершена. Лучшая конфигурация: {best_name}")


if __name__ == "__main__":
    main()
