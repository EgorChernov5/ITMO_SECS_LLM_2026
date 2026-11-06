"""Оценивает retrieval для экспериментальных конфигураций."""

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from source.configuration import (
    get_experiment_config,
    get_experiment_retrieval_config,
    load_pipeline_config,
)
from source.constants import (
    CONFIG_PATH,
    QUESTIONS_PATH,
    RESULTS_DIR,
    RESULTS_RUNS_DIR,
)
from source.embedding import DenseEmbedder
from source.retrieval import create_retriever
from source.scripts.rag import ask_ollama, build_prompt
from source.structures import (
    EvaluationQuestion,
    ExperimentConfig,
    RetrievalConfig,
)
from source.vector_store import (
    collection_name,
    create_qdrant_client,
    load_latest_build,
    qdrant_settings,
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
    parser.add_argument("--experiment", required=True)
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

        raw_chunks = raw.get("relevant_chunks")
        if (
            not isinstance(raw_chunks, list)
            or not raw_chunks
            or any(
                not isinstance(chunk, str) or not chunk.strip()
                for chunk in raw_chunks
            )
        ):
            raise ValueError(
                "У каждого вопроса должен быть непустой список "
                "relevant_chunks."
            )
        chunks = tuple(chunk.strip() for chunk in raw_chunks)

        raw_facts = raw.get("reference_facts")
        if not isinstance(raw_facts, list) or not raw_facts:
            raise ValueError(
                "У каждого вопроса должен быть непустой список "
                "reference_facts."
            )
        facts: list[tuple[str, ...]] = []
        for fact in raw_facts:
            if (
                not isinstance(fact, list)
                or not fact
                or any(
                    not isinstance(variant, str) or not variant.strip()
                    for variant in fact
                )
            ):
                raise ValueError(
                    "Каждый эталонный факт должен содержать непустой список "
                    "допустимых формулировок."
                )
            facts.append(tuple(variant.strip() for variant in fact))

        questions.append(
            EvaluationQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]).strip(),
                relevant_pages=pages,
                relevant_chunks=chunks,
                reference_facts=tuple(facts),
            )
        )
    if len({question.id for question in questions}) != len(questions):
        raise ValueError("Идентификаторы вопросов должны быть уникальными.")
    return questions


def normalize_evaluation_text(text: str) -> str:
    """
    Нормализует текст для детерминированного поиска эталонных фактов.

    Parameters:
        text (str): Исходный ответ или допустимая формулировка факта.

    Returns:
        str: Текст в нижнем регистре без пунктуации и лишних пробелов.

    Fallbacks:
        Буква «ё» заменяется на «е».
    """
    normalized = re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е"))
    return " ".join(normalized.split())


def calculate_metrics(
    results: list[dict[str, object]],
    relevant_pages: tuple[int, ...],
    k: int,
    generated_answer: str | None = None,
    reference_facts: tuple[tuple[str, ...], ...] = (),
) -> dict[str, float]:
    """
    Вычисляет retrieval-метрики и полноту сгенерированного ответа.

    Parameters:
        results (list[dict[str, object]]): Ранжированные результаты поиска.
        relevant_pages (tuple[int, ...]): Эталонные страницы.
        k (int): Глубина оценки.
        generated_answer (str | None): Ответ RAG для оценки полноты.
            По умолчанию: None.
        reference_facts (tuple[tuple[str, ...], ...]): Допустимые
            формулировки обязательных фактов. По умолчанию: пустой кортеж.

    Returns:
        dict[str, float]: Значения precision, recall, reciprocal_rank и
        fact_recall при наличии ответа.

    Fallbacks:
        При отсутствии результатов возвращает нулевые retrieval-метрики.
        Fact Recall не добавляется без ответа или эталонных фактов.
    """
    expected = set(relevant_pages)
    retrieved_pages: set[int] = set()
    relevant_results = 0
    reciprocal_rank = 0.0
    for rank, result in enumerate(results[:k], start=1):
        payload = result.get("payload", {})
        if not isinstance(payload, dict):
            continue
        page_start = int(payload.get("page_start", 0))
        page_end = int(payload.get("page_end", page_start))
        matched = expected.intersection(range(page_start, page_end + 1))
        if matched:
            relevant_results += 1
            retrieved_pages.update(matched)
            if reciprocal_rank == 0:
                reciprocal_rank = 1 / rank

    metrics = {
        "precision": relevant_results / k,
        "recall": len(retrieved_pages) / len(expected),
        "reciprocal_rank": reciprocal_rank,
    }
    if generated_answer is not None and reference_facts:
        normalized_answer = normalize_evaluation_text(generated_answer)
        matched_facts = sum(
            any(
                f" {normalize_evaluation_text(variant)} "
                in f" {normalized_answer} "
                for variant in variants
            )
            for variants in reference_facts
        )
        metrics["fact_recall"] = matched_facts / len(reference_facts)
    return metrics


def evaluate_config(
    client: object,
    settings: dict[str, object],
    config_name: str,
    experiment: ExperimentConfig,
    retrieval: RetrievalConfig,
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
        experiment (ExperimentConfig): Проверенный эксперимент.
        retrieval (RetrievalConfig): Разрешённая retrieval-конфигурация.
        questions (list[EvaluationQuestion]): Оценочные вопросы.
        vectors (object): Матрица эмбеддингов вопросов.
        warmup_runs (int): Число прогревочных запросов на вопрос.
        repetitions (int): Число измеряемых запросов на вопрос.

    Returns:
        dict[str, object]: Детальные и агрегированные метрики.

    Fallbacks:
        Оценка не начинается до полной готовности HNSW.
    """
    _, manifest, _, _ = load_latest_build(config_name)
    qdrant = qdrant_settings(settings)
    wait_for_collection_index(
        client,
        collection_name(settings, config_name),
        int(manifest["chunk_count"]),
        float(qdrant.get("indexing_wait_timeout_seconds", 180)),
    )
    retriever = create_retriever(
        client,
        collection_name(settings, config_name),
        experiment.ef_search,
        retrieval,
    )
    per_question: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    generation_latencies_ms: list[float] = []

    for index, question in enumerate(questions):
        vector = vectors[index].tolist()
        for _ in range(warmup_runs):
            retriever.retrieve(question.question, vector)

        results: list[dict[str, object]] = []
        question_latencies: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            current_results = retriever.retrieve(question.question, vector)
            elapsed_ms = (time.perf_counter() - started) * 1000
            question_latencies.append(elapsed_ms)
            latencies_ms.append(elapsed_ms)
            if not results:
                results = current_results

        generation_started = time.perf_counter()
        answer = ask_ollama(
            settings,
            build_prompt(question.question, results),
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        generation_latencies_ms.append(generation_ms)

        at_k = calculate_metrics(
            results,
            question.relevant_pages,
            retrieval.top_k,
            generated_answer=answer,
            reference_facts=question.reference_facts,
        )
        per_question.append(
            {
                "id": question.id,
                "question": question.question,
                "relevant_pages": list(question.relevant_pages),
                "relevant_chunks": list(question.relevant_chunks),
                "reference_facts": [
                    list(fact) for fact in question.reference_facts
                ],
                "generated_answer": answer,
                "retrieved": results,
                "precision_at_k": at_k["precision"],
                "recall_at_k": at_k["recall"],
                "reciprocal_rank": at_k["reciprocal_rank"],
                "fact_recall": at_k["fact_recall"],
                "median_search_ms": median(question_latencies),
                "generation_ms": generation_ms,
            }
        )

    count = len(per_question)
    aggregate = {
        key: sum(float(item[key]) for item in per_question) / count
        for key in (
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "fact_recall",
        )
    }
    aggregate["average_search_ms"] = mean(latencies_ms)
    aggregate["median_search_ms"] = median(latencies_ms)
    aggregate["average_generation_ms"] = mean(generation_latencies_ms)
    aggregate["median_generation_ms"] = median(generation_latencies_ms)
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
            "retrieval": retrieval.name,
            "bi_encoder": retrieval.bi_encoder.model,
            "bi_encoder_id": retrieval.bi_encoder.name,
            "cross_encoder": (
                retrieval.cross_encoder.model if retrieval.cross_encoder else None
            ),
            "cross_encoder_id": (
                retrieval.cross_encoder.name if retrieval.cross_encoder else None
            ),
            "candidate_top_k": retrieval.candidate_top_k,
            "top_k": retrieval.top_k,
        },
        "costs": {
            "chunk_count": manifest["chunk_count"],
            "chunking_seconds": manifest["chunking_seconds"],
            "embedding_seconds": manifest["embedding_seconds"],
        },
        "aggregate": aggregate,
        "questions": per_question,
    }


def select_best(results: list[dict[str, object]]) -> dict[str, object]:
    """
    Выбирает лучшую конфигурацию по retrieval-метрикам.

    Parameters:
        results (list[dict[str, object]]): Результаты всех экспериментов.

    Returns:
        dict[str, object]: Результат лучшей конфигурации.

    Fallbacks:
        При пустом списке возбуждает ValueError.
    """
    if not results:
        raise ValueError("Нет результатов для выбора лучшей конфигурации.")
    return max(
        results,
        key=lambda result: (
            float(result["aggregate"]["reciprocal_rank"]),
            float(result["aggregate"]["recall_at_k"]),
            -float(result["aggregate"]["median_search_ms"]),
        ),
    )


def render_table(
    results: list[dict[str, object]],
    highlight_best: bool,
) -> list[str]:
    """
    Формирует Markdown-таблицу результатов конфигураций.

    Parameters:
        results (list[dict[str, object]]): Результаты конфигураций.
        highlight_best (bool): Признак выделения лучших значений.

    Returns:
        list[str]: Строки Markdown-таблицы.

    Fallbacks:
        При равенстве выделяет все лучшие значения.
    """
    # Определяем экстремумы по исходным значениям до округления.
    aggregates = [result["aggregate"] for result in results]
    costs = [result["costs"] for result in results]
    best_values = (
        max(float(value["precision_at_k"]) for value in aggregates),
        max(float(value["recall_at_k"]) for value in aggregates),
        max(float(value["reciprocal_rank"]) for value in aggregates),
        max(float(value["fact_recall"]) for value in aggregates),
        min(float(value["median_search_ms"]) for value in aggregates),
        min(float(value["median_generation_ms"]) for value in aggregates),
        min(int(value["chunk_count"]) for value in costs),
    )
    rows = [
        (
            "| Конфигурация | K | P@K | R@K | MRR | Fact Recall | "
            "Retrieval, мс | Генерация, мс | Чанки |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        # Форматируем значения одной конфигурации и выделяем совпавшие экстремумы.
        aggregate = result["aggregate"]
        costs = result["costs"]
        parameters = result.get("parameters", {})
        top_k = int(parameters.get("top_k", 5))
        values = (
            float(aggregate["precision_at_k"]),
            float(aggregate["recall_at_k"]),
            float(aggregate["reciprocal_rank"]),
            float(aggregate["fact_recall"]),
            float(aggregate["median_search_ms"]),
            float(aggregate["median_generation_ms"]),
            int(costs["chunk_count"]),
        )
        formatted = [
            f"{value:.4f}" if index < 4 else f"{value:.2f}"
            for index, value in enumerate(values[:-1])
        ]
        formatted.append(str(values[-1]))
        if highlight_best:
            formatted = [
                f"**{text}**" if value == best else text
                for text, value, best in zip(formatted, values, best_values)
            ]
        rows.append(
            f"| {result['config']} | {top_k} | "
            + " | ".join(formatted)
            + " |"
        )
    return rows


def render_report(results: list[dict[str, object]]) -> str:
    """
    Формирует локальный Markdown-отчёт запуска.

    Parameters:
        results (list[dict[str, object]]): Результаты конфигураций.

    Returns:
        str: Полный Markdown-отчёт.

    Fallbacks:
        Не предусмотрены.
    """
    # Сохраняем прежний формат отчёта и локальный выбор конфигурации.
    best_name = str(select_best(results)["config"])
    rows = [
        "# Сравнение конфигураций retrieval и генерации",
        "",
        *render_table(results, False),
    ]
    rows.extend(
        [
            "",
            "## Вывод",
            "",
            f"Лучшая конфигурация: **{best_name}**. "
            "Выбор выполнен по MRR, затем Recall@K и медиане времени retrieval.",
            "",
            "Время retrieval измерено на клиенте, включает задержку "
            "SSH-туннеля и Cross-Encoder reranking при его наличии.",
            "",
            "## Качество генерации",
            "",
            "Fact Recall показывает долю обязательных эталонных фактов, "
            "обнаруженных в нормализованном ответе. Метрика оценивает "
            "полноту, но не выявляет неподтверждённые утверждения.",
            "",
        ]
    )
    return "\n".join(rows)


def load_runs() -> list[dict[str, object]]:
    """
    Загружает сохранённые завершённые запуски.

    Parameters:
        Не предусмотрены.

    Returns:
        list[dict[str, object]]: Запуски в хронологическом порядке.

    Fallbacks:
        Повреждённый результат останавливает пересборку leaderboard.
    """
    runs: list[dict[str, object]] = []
    if not RESULTS_RUNS_DIR.is_dir():
        return runs

    # Читаем только финальные каталоги внутри runs.
    for run_dir in sorted(path for path in RESULTS_RUNS_DIR.iterdir() if path.is_dir()):
        result_path = run_dir / "result.json"
        report_path = run_dir / "report.md"
        if not result_path.is_file() or not report_path.is_file():
            raise ValueError(f"Незавершённый запуск: {run_dir}.")
        try:
            results = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Не удалось прочитать {result_path}: {error}") from error
        if not isinstance(results, list) or not results:
            raise ValueError(f"Результат {result_path} должен быть непустым списком.")
        # Нормализуем старые @5 поля в единый формат текущего leaderboard.
        for result in results:
            aggregate = result["aggregate"]
            if "precision_at_k" not in aggregate:
                aggregate["precision_at_k"] = aggregate["precision_at_5"]
            if "recall_at_k" not in aggregate:
                aggregate["recall_at_k"] = aggregate["recall_at_5"]
            parameters = result.setdefault("parameters", {})
            parameters.setdefault("top_k", 5)
            for question in result.get("questions", []):
                if "precision_at_k" not in question:
                    question["precision_at_k"] = question["precision_at_5"]
                if "recall_at_k" not in question:
                    question["recall_at_k"] = question["recall_at_5"]
        _, separator, experiment = run_dir.name.partition("_")
        if not separator or not experiment:
            raise ValueError(f"Некорректный идентификатор запуска: {run_dir.name}.")
        configurations = [
            {
                "config": result["config"],
                "parameters": result["parameters"],
                "aggregate": result["aggregate"],
                "costs": result["costs"],
            }
            for result in results
        ]
        runs.append(
            {
                "run_id": run_dir.name,
                "experiment": experiment,
                "result": f"runs/{run_dir.name}/result.json",
                "report": f"runs/{run_dir.name}/report.md",
                "best_config": str(select_best(results)["config"]),
                "configurations": configurations,
            }
        )
    return runs


def rebuild_leaderboard() -> None:
    """
    Пересобирает сводные отчёты и указатели запусков.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Записывает файлы в каталог результатов.

    Fallbacks:
        При отсутствии запусков возбуждает ValueError.
    """
    runs = load_runs()
    if not runs:
        raise ValueError("Нет завершённых запусков для leaderboard.")

    # Сохраняем компактный машиночитаемый индекс запусков.
    (RESULTS_DIR / "leaderboard.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Формируем отдельный раздел и таблицу для каждого запуска.
    rows = ["# Сравнение экспериментов", ""]
    for run in runs:
        rows.extend(
            [
                f"## {run['experiment']}",
                "",
                f"Запуск: `{run['run_id']}`",
                "",
                *render_table(run["configurations"], True),
                "",
            ]
        )
    (RESULTS_DIR / "leaderboard.md").write_text(
        "\n".join(rows),
        encoding="utf-8",
        newline="\n",
    )

    # Обновляем указатель на последний хронологический запуск.
    latest = runs[-1]
    latest_pointer = {
        key: latest[key]
        for key in ("run_id", "experiment", "result", "report")
    }
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(latest_pointer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Выбираем лучшую конфигурацию среди всех сохранённых запусков.
    candidates = [
        (run, configuration)
        for run in runs
        for configuration in run["configurations"]
    ]
    best_run, best_config = max(
        candidates,
        key=lambda item: (
            float(item[1]["aggregate"]["reciprocal_rank"]),
            float(item[1]["aggregate"]["recall_at_k"]),
            -float(item[1]["aggregate"]["median_search_ms"]),
        ),
    )
    best_pointer = {
        "run_id": best_run["run_id"],
        "experiment": best_run["experiment"],
        "config": best_config["config"],
    }
    (RESULTS_DIR / "best_experiment.json").write_text(
        json.dumps(best_pointer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def save_results(
    results: list[dict[str, object]],
    experiment: str,
) -> tuple[str, str]:
    """
    Сохраняет неизменяемый запуск и обновляет сводные результаты.

    Parameters:
        results (list[dict[str, object]]): Результаты конфигураций.
        experiment (str): Название эксперимента из CLI.

    Returns:
        tuple[str, str]: Идентификатор запуска и лучшая конфигурация.

    Fallbacks:
        Не перезаписывает существующий идентификатор запуска.
    """
    # Нормализуем название для безопасного имени каталога.
    slug = re.sub(r"[^\w-]+", "_", experiment.strip()).strip("_-")
    if not slug:
        raise ValueError("Название эксперимента не должно быть пустым.")
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{slug}"
    run_dir = RESULTS_RUNS_DIR / run_id
    pending_dir = RESULTS_DIR / f".{run_id}.pending"
    if run_dir.exists() or pending_dir.exists():
        raise FileExistsError(f"Запуск уже существует: {run_id}.")

    # Готовим оба артефакта до публикации запуска в каталоге runs.
    best_name = str(select_best(results)["config"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir()
    (pending_dir / "result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (pending_dir / "report.md").write_text(
        render_report(results),
        encoding="utf-8",
        newline="\n",
    )
    pending_dir.rename(run_dir)

    # Пересобираем производные файлы после публикации нового запуска.
    rebuild_leaderboard()
    return run_id, best_name


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
    client = create_qdrant_client(settings)
    results: list[dict[str, object]] = []
    for config_name in args.configs:
        # Векторизуем вопросы моделью конкретной конфигурации без смешения векторов.
        experiment = get_experiment_config(settings, config_name)
        retrieval = get_experiment_retrieval_config(settings, experiment)
        embedder = DenseEmbedder(retrieval.bi_encoder.model)
        vectors = embedder.embed(
            [question.question for question in questions],
            batch_size=retrieval.bi_encoder.batch_size,
        )
        results.append(
            evaluate_config(
                client,
                settings,
                config_name,
                experiment,
                retrieval,
                questions,
                vectors,
                warmup_runs,
                repetitions,
            )
        )
    run_id, best_name = save_results(results, args.experiment)
    print(
        f"Оценка завершена. Запуск: {run_id}. "
        f"Лучшая конфигурация: {best_name}"
    )


if __name__ == "__main__":
    main()
