"""Рассчитывает статистику рецептов и реальных RAG-промптов."""

import argparse
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from source.configuration import get_experiment_config, load_pipeline_config
from source.constants import (
    CONFIG_PATH,
    DEFAULT_CONFIG_NAME,
    EXPLORED_DIR,
    MODEL_NAME,
    PAGE_MARKER_PATTERN,
    QUESTIONS_PATH,
    QWEN_TOKENIZER_NAME,
    RECIPES_PATH,
)
from source.embedding import DenseEmbedder
from source.scripts.build_index import create_indexing_strategy
from source.scripts.evaluate_retrieval import load_questions
from source.scripts.rag import RAG_INSTRUCTION, build_prompt
from source.structures import GenerationPromptMetrics, RecipeMetrics
from source.vector_store import collection_name, create_qdrant_client, search_chunks


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры анализа рецептов и generation-промптов.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы командной строки.

    Fallbacks:
        По умолчанию выполняет оба вида анализа для baseline-конфигурации.
    """
    # Настраиваем источники, модели и выбранную ветку анализа.
    parser = argparse.ArgumentParser(
        description="Расчёт статистики рецептов и RAG-промптов."
    )
    parser.add_argument("--recipes-path", type=Path, default=RECIPES_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPLORED_DIR)
    parser.add_argument("--experiment", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--settings", type=Path, default=CONFIG_PATH)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--sbert-tokenizer")
    parser.add_argument("--qwen-tokenizer", default=QWEN_TOKENIZER_NAME)
    parser.add_argument(
        "--analysis-mode",
        choices=("both", "retrieval", "generation"),
        default="both",
    )
    return parser.parse_args()


def analyze_recipes(
    recipes_path: Path = RECIPES_PATH,
    output_dir: Path = EXPLORED_DIR,
    experiment: str = DEFAULT_CONFIG_NAME,
    settings_path: Path = CONFIG_PATH,
    sbert_tokenizer_name: str | None = None,
    qwen_tokenizer_name: str = QWEN_TOKENIZER_NAME,
    questions_path: Path = QUESTIONS_PATH,
    analysis_mode: str = "both",
) -> tuple[Path, ...]:
    """
    Рассчитывает показатели рецептов и фактических RAG-промптов.

    Parameters:
        recipes_path (Path): Путь к Markdown с рецептами. По умолчанию: RECIPES_PATH.
        output_dir (Path): Каталог результатов. По умолчанию: EXPLORED_DIR.
        experiment (str): Конфигурация чанкинга и поиска. По умолчанию: baseline.
        settings_path (Path): Путь к настройкам пайплайна. По умолчанию: CONFIG_PATH.
        sbert_tokenizer_name (str | None): Модель SBERT. По умолчанию: из настроек.
        qwen_tokenizer_name (str): Токенизатор Qwen. По умолчанию: Qwen2.5-3B-Instruct.
        questions_path (Path): Путь к оценочным вопросам. По умолчанию: QUESTIONS_PATH.
        analysis_mode (str): Режим both, retrieval или generation. По умолчанию: both.

    Returns:
        tuple[Path, ...]: Пути только к таблицам выбранного режима.

    Fallbacks:
        Retrieval сохраняет прежние таблицы, generation требует доступных GPU и Qdrant.
    """
    # Проверяем режим, входной Markdown и название раздела эксперимента.
    if analysis_mode not in {"both", "retrieval", "generation"}:
        raise ValueError("Режим анализа должен быть both, retrieval или generation.")
    if not recipes_path.is_file():
        raise FileNotFoundError(f"Файл с рецептами не найден: {recipes_path}.")
    markdown = recipes_path.read_text(encoding="utf-8")
    if not markdown.strip():
        raise ValueError("Файл с рецептами пуст.")
    if not experiment.strip() or "\n" in experiment or "\r" in experiment:
        raise ValueError(
            "Название эксперимента должно быть непустой строкой без переносов."
        )

    # Один раз загружаем общие настройки эксперимента и токенизатор Qwen.
    settings = load_pipeline_config(settings_path)
    experiment_config = get_experiment_config(settings, experiment)
    embedding = settings.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("Секция embedding должна быть JSON-объектом.")
    sbert_name = sbert_tokenizer_name or str(embedding.get("model", MODEL_NAME))
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_name)
    result_paths: list[Path] = []

    # В generation загружаем полноценный эмбеддер, в retrieval — только токенизатор.
    if analysis_mode in {"both", "generation"}:
        embedder = DenseEmbedder(sbert_name)
        sbert_tokenizer = embedder.tokenizer
    else:
        sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_name)
        embedder = DenseEmbedder.__new__(DenseEmbedder)
        embedder.model_name = sbert_name
        embedder.tokenizer = sbert_tokenizer

    # Анализ использует то же выделение физических чанков, что и индексатор.
    strategy = create_indexing_strategy(experiment_config, embedder, 8)

    # Retrieval-ветка сохраняет прежний расчёт показателей каждого рецепта.
    if analysis_mode in {"both", "retrieval"}:
        recipe_matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
        category_matches = list(re.finditer(r"(?m)^#\s+(.+?)\s*$", markdown))
        if not recipe_matches:
            raise ValueError(
                "В Markdown не найдены рецепты с заголовками второго уровня."
            )
        metrics: list[RecipeMetrics] = []
        for index, recipe_match in enumerate(recipe_matches):
            next_recipe = (
                recipe_matches[index + 1].start()
                if index + 1 < len(recipe_matches)
                else len(markdown)
            )
            next_category = next(
                (
                    match.start()
                    for match in category_matches
                    if recipe_match.end() < match.start() < next_recipe
                ),
                next_recipe,
            )
            category = next(
                (
                    match.group(1).strip()
                    for match in reversed(category_matches)
                    if match.start() < recipe_match.start()
                ),
                "Без категории",
            )
            recipe_text = re.sub(
                PAGE_MARKER_PATTERN,
                "",
                markdown[recipe_match.start() : next_category],
            ).strip()

            # Считаем параметры полного текста и чанки штатным алгоритмом проекта.
            chunks = strategy.create_chunks(f"# {category}\n\n{recipe_text}")
            metrics.append(
                RecipeMetrics(
                    category=category,
                    recipe=recipe_match.group(1).strip(),
                    sbert_tokens=len(
                        sbert_tokenizer.encode(recipe_text, add_special_tokens=False)
                    ),
                    qwen_tokens=len(
                        qwen_tokenizer.encode(recipe_text, add_special_tokens=False)
                    ),
                    chunk_count=len(chunks),
                    character_count=len(recipe_text),
                    word_count=len(re.findall(r"[^\W\d_]+", recipe_text)),
                )
            )

        # Формируем детальную retrieval-таблицу и статистику по показателям.
        details = pd.DataFrame(asdict(item) for item in metrics)
        summary = pd.DataFrame(
            {
                "parameter": parameter,
                "mean": details[parameter].mean(),
                "median": details[parameter].median(),
                "std": details[parameter].std(ddof=0),
                "minimum": details[parameter].min(),
                "maximum": details[parameter].max(),
            }
            for parameter in (
                "sbert_tokens",
                "qwen_tokens",
                "chunk_count",
                "character_count",
                "word_count",
            )
        )

        # Сохраняем CSV и готовим новую таблицу раздела эксперимента.
        output_dir.mkdir(parents=True, exist_ok=True)
        details_path = output_dir / "recipes_metrics.csv"
        summary_path = output_dir / "recipes_metrics_summary.md"
        details.to_csv(details_path, index=False, encoding="utf-8", lineterminator="\n")
        table = "\n".join(
            [
                "| parameter | mean | median | std | minimum | maximum |",
                "|---|---:|---:|---:|---:|---:|",
                *(
                    "| {parameter} | {mean:.2f} | {median:.2f} | {std:.2f} "
                    "| {minimum:.0f} | {maximum:.0f} |".format(**row)
                    for row in summary.to_dict(orient="records")
                ),
            ]
        )
        section = f"## {experiment}\n\n{table}\n"

        # Обновляем только таблицу retrieval, сохраняя ручные заметки раздела.
        summary_markdown = (
            summary_path.read_text(encoding="utf-8")
            if summary_path.is_file()
            else "# Сводная статистика рецептов\n"
        )
        section_pattern = re.compile(
            rf"(?ms)^## {re.escape(experiment)}[ \t]*$.*?(?=^## |\Z)"
        )
        table_pattern = re.compile(
            r"(?m)^\| parameter \| mean \| median \| std \| minimum \| maximum \|"
            r"[ \t]*\r?\n^\|---\|---:\|---:\|---:\|---:\|---:\|[ \t]*$"
            r"(?:\r?\n^\|.*\|[ \t]*$)*"
        )
        section_match = section_pattern.search(summary_markdown)
        if section_match:
            current_section = section_match.group()
            updated_section = (
                table_pattern.sub(table, current_section)
                if table_pattern.search(current_section)
                else f"{current_section.rstrip()}\n\n{table}\n"
            )
            summary_markdown = (
                summary_markdown[: section_match.start()]
                + updated_section
                + summary_markdown[section_match.end() :]
            )
        else:
            summary_markdown = f"{summary_markdown.rstrip()}\n\n{section}"
        summary_path.write_text(summary_markdown, encoding="utf-8", newline="\n")
        result_paths.extend((details_path, summary_path))

    # Generation-ветка один раз создаёт штатные чанки и загружает вопросы.
    if analysis_mode in {"both", "generation"}:
        chunks = strategy.create_chunks(markdown)
        if not chunks:
            raise ValueError("Не удалось получить ни одного чанка из Markdown.")
        questions = load_questions(questions_path)
        evaluation = settings.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValueError("Секция evaluation должна быть JSON-объектом.")
        generation_top_k = int(evaluation.get("generation_top_k", 5))
        batch_size = int(embedding.get("batch_size", 8))
        if generation_top_k <= 0:
            raise ValueError("Число чанков для генерации должно быть положительным.")
        if batch_size <= 0:
            raise ValueError("Размер пакета должен быть положительным.")

        # Получаем векторы вопросов и выполняем тот же поиск Qdrant, что основной RAG.
        vectors = embedder.embed(
            [question.question for question in questions],
            batch_size=batch_size,
        )
        client = create_qdrant_client(settings)
        name = collection_name(settings, experiment)
        instruction_tokens = len(
            qwen_tokenizer.encode(RAG_INSTRUCTION, add_special_tokens=False)
        )
        generation_metrics: list[GenerationPromptMetrics] = []
        for question, vector in zip(questions, vectors):
            results = search_chunks(
                client,
                name,
                vector.tolist(),
                limit=10,
                ef_search=experiment_config.ef_search,
            )[:generation_top_k]
            prompt = build_prompt(question.question, results)
            context = "\n\n".join(
                str(result["payload"]["text"]) for result in results
            )
            generation_metrics.append(
                GenerationPromptMetrics(
                    question_id=question.id,
                    question=question.question,
                    context_chunk_count=len(results),
                    instruction_tokens=instruction_tokens,
                    context_tokens=len(
                        qwen_tokenizer.encode(context, add_special_tokens=False)
                    ),
                    question_tokens=len(
                        qwen_tokenizer.encode(
                            question.question,
                            add_special_tokens=False,
                        )
                    ),
                    prompt_tokens=len(
                        qwen_tokenizer.encode(prompt, add_special_tokens=False)
                    ),
                    character_count=len(prompt),
                    word_count=len(re.findall(r"[^\W\d_]+", prompt)),
                )
            )

        # Формируем отдельные generation-таблицы с прежним набором статистик.
        generation_details = pd.DataFrame(
            asdict(item) for item in generation_metrics
        )
        generation_summary = pd.DataFrame(
            {
                "parameter": parameter,
                "mean": generation_details[parameter].mean(),
                "median": generation_details[parameter].median(),
                "std": generation_details[parameter].std(ddof=0),
                "minimum": generation_details[parameter].min(),
                "maximum": generation_details[parameter].max(),
            }
            for parameter in (
                "context_chunk_count",
                "instruction_tokens",
                "context_tokens",
                "question_tokens",
                "prompt_tokens",
                "character_count",
                "word_count",
            )
        )

        # Сохраняем generation CSV и готовим таблицу выбранного эксперимента.
        output_dir.mkdir(parents=True, exist_ok=True)
        generation_details_path = output_dir / "generation_prompt_metrics.csv"
        generation_summary_path = (
            output_dir / "generation_prompt_metrics_summary.md"
        )
        generation_details.to_csv(
            generation_details_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        generation_table = "\n".join(
            [
                "| parameter | mean | median | std | minimum | maximum |",
                "|---|---:|---:|---:|---:|---:|",
                *(
                    "| {parameter} | {mean:.2f} | {median:.2f} | {std:.2f} "
                    "| {minimum:.0f} | {maximum:.0f} |".format(**row)
                    for row in generation_summary.to_dict(orient="records")
                ),
            ]
        )
        generation_section = f"## {experiment}\n\n{generation_table}\n"

        # Обновляем только generation-таблицу, сохраняя ручные заметки раздела.
        generation_markdown = (
            generation_summary_path.read_text(encoding="utf-8")
            if generation_summary_path.is_file()
            else "# Сводная статистика generation-промптов\n"
        )
        generation_section_pattern = re.compile(
            rf"(?ms)^## {re.escape(experiment)}[ \t]*$.*?(?=^## |\Z)"
        )
        generation_table_pattern = re.compile(
            r"(?m)^\| parameter \| mean \| median \| std \| minimum \| maximum \|"
            r"[ \t]*\r?\n^\|---\|---:\|---:\|---:\|---:\|---:\|[ \t]*$"
            r"(?:\r?\n^\|.*\|[ \t]*$)*"
        )
        generation_section_match = generation_section_pattern.search(
            generation_markdown
        )
        if generation_section_match:
            current_section = generation_section_match.group()
            updated_section = (
                generation_table_pattern.sub(generation_table, current_section)
                if generation_table_pattern.search(current_section)
                else f"{current_section.rstrip()}\n\n{generation_table}\n"
            )
            generation_markdown = (
                generation_markdown[: generation_section_match.start()]
                + updated_section
                + generation_markdown[generation_section_match.end() :]
            )
        else:
            generation_markdown = (
                f"{generation_markdown.rstrip()}\n\n{generation_section}"
            )
        generation_summary_path.write_text(
            generation_markdown,
            encoding="utf-8",
            newline="\n",
        )
        result_paths.extend((generation_details_path, generation_summary_path))

    return tuple(result_paths)


def main() -> None:
    """
    Запускает выбранный режим анализа из командной строки.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Печатает пути только к созданным таблицам.

    Fallbacks:
        По умолчанию создаёт retrieval- и generation-таблицы за один запуск.
    """
    # Выполняем анализ с параметрами CLI и показываем все созданные пары файлов.
    args = parse_args()
    paths = analyze_recipes(
        recipes_path=args.recipes_path,
        output_dir=args.output_dir,
        experiment=args.experiment,
        settings_path=args.settings,
        sbert_tokenizer_name=args.sbert_tokenizer,
        qwen_tokenizer_name=args.qwen_tokenizer,
        questions_path=args.questions,
        analysis_mode=args.analysis_mode,
    )
    for index in range(0, len(paths), 2):
        print(f"Детальная таблица: {paths[index]}")
        print(f"Сводная таблица: {paths[index + 1]}")


if __name__ == "__main__":
    main()
