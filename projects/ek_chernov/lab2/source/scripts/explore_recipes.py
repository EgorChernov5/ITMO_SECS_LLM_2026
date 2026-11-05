"""Рассчитывает детальную и сводную статистику по рецептам."""

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
    QWEN_TOKENIZER_NAME,
    RECIPES_PATH,
)
from source.embedding import DenseEmbedder
from source.scripts.build_index import create_chunks, parse_markdown_units
from source.structures import RecipeMetrics


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры анализа рецептов.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы командной строки.

    Fallbacks:
        По умолчанию анализирует recipes_2.md с baseline-конфигурацией.
    """
    # Настраиваем пути, конфигурацию чанкинга и токенизаторы.
    parser = argparse.ArgumentParser(
        description="Расчёт статистики токенов, чанков, символов и слов."
    )
    parser.add_argument("--recipes-path", type=Path, default=RECIPES_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPLORED_DIR)
    parser.add_argument("--experiment", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--settings", type=Path, default=CONFIG_PATH)
    parser.add_argument("--sbert-tokenizer")
    parser.add_argument("--qwen-tokenizer", default=QWEN_TOKENIZER_NAME)
    return parser.parse_args()


def analyze_recipes(
    recipes_path: Path = RECIPES_PATH,
    output_dir: Path = EXPLORED_DIR,
    experiment: str = DEFAULT_CONFIG_NAME,
    settings_path: Path = CONFIG_PATH,
    sbert_tokenizer_name: str | None = None,
    qwen_tokenizer_name: str = QWEN_TOKENIZER_NAME,
) -> tuple[Path, Path]:
    """
    Рассчитывает показатели рецептов и сохраняет CSV и Markdown-таблицу.

    Parameters:
        recipes_path (Path): Путь к Markdown с рецептами. По умолчанию: RECIPES_PATH.
        output_dir (Path): Каталог результатов. По умолчанию: EXPLORED_DIR.
        experiment (str): Конфигурация чанкинга и название раздела. По умолчанию: baseline.
        settings_path (Path): Путь к настройкам пайплайна. По умолчанию: CONFIG_PATH.
        sbert_tokenizer_name (str | None): Токенизатор SBERT. По умолчанию: из настроек.
        qwen_tokenizer_name (str): Токенизатор Qwen. По умолчанию: Qwen2.5-3B-Instruct.

    Returns:
        tuple[Path, Path]: Пути к детальной CSV и сводной Markdown-таблице.

    Fallbacks:
        При пустом файле или отсутствии рецептов возбуждает ValueError.
    """
    # Загружаем текст, параметры чанкинга и названия моделей.
    if not recipes_path.is_file():
        raise FileNotFoundError(f"Файл с рецептами не найден: {recipes_path}.")
    markdown = recipes_path.read_text(encoding="utf-8")
    if not markdown.strip():
        raise ValueError("Файл с рецептами пуст.")
    settings = load_pipeline_config(settings_path)
    if not experiment.strip() or "\n" in experiment or "\r" in experiment:
        raise ValueError(
            "Название эксперимента должно быть непустой строкой без переносов."
        )
    experiment_config = get_experiment_config(settings, experiment)
    embedding = settings.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("Секция embedding должна быть JSON-объектом.")
    sbert_name = sbert_tokenizer_name or str(embedding.get("model", MODEL_NAME))

    # Загружаем только токенизаторы, не создавая модели на GPU.
    sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_name)
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_name)
    embedder = DenseEmbedder.__new__(DenseEmbedder)
    embedder.model_name = sbert_name
    embedder.tokenizer = sbert_tokenizer

    # Находим каждый рецепт и ближайшую предшествующую категорию.
    recipe_matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    category_matches = list(re.finditer(r"(?m)^#\s+(.+?)\s*$", markdown))
    if not recipe_matches:
        raise ValueError("В Markdown не найдены рецепты с заголовками второго уровня.")
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
        chunk_markdown = f"# {category}\n\n{recipe_text}"
        chunks = create_chunks(
            parse_markdown_units(chunk_markdown),
            experiment_config,
            embedder,
        )
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

    # Формируем детальную таблицу и генеральную статистику по показателям.
    details = pd.DataFrame(asdict(item) for item in metrics)
    parameter_columns = [
        "sbert_tokens",
        "qwen_tokens",
        "chunk_count",
        "character_count",
        "word_count",
    ]
    summary = pd.DataFrame(
        {
            "parameter": parameter,
            "mean": details[parameter].mean(),
            "median": details[parameter].median(),
            "std": details[parameter].std(ddof=0),
            "minimum": details[parameter].min(),
            "maximum": details[parameter].max(),
        }
        for parameter in parameter_columns
    )

    # Сохраняем детальные результаты и формируем Markdown-таблицу эксперимента.
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "recipes_metrics.csv"
    summary_path = output_dir / "recipes_metrics_summary.md"
    details.to_csv(details_path, index=False, encoding="utf-8", lineterminator="\n")
    table_lines = [
        "| parameter | mean | median | std | minimum | maximum |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            "| {parameter} | {mean:.2f} | {median:.2f} | {std:.2f} "
            "| {minimum:.0f} | {maximum:.0f} |".format(**row)
            for row in summary.to_dict(orient="records")
        ),
    ]
    table = "\n".join(table_lines)
    section = f"## {experiment}\n\n{table}\n"

    # Читаем существующую сводку, чтобы сохранить результаты других экспериментов.
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

    # Обновляем только таблицу раздела, не удаляя добавленные вручную заметки.
    if section_match:
        current_section = section_match.group()
        if table_pattern.search(current_section):
            updated_section = table_pattern.sub(table, current_section)
        else:
            updated_section = f"{current_section.rstrip()}\n\n{table}\n"
        summary_markdown = (
            summary_markdown[: section_match.start()]
            + updated_section
            + summary_markdown[section_match.end() :]
        )
    else:
        summary_markdown = f"{summary_markdown.rstrip()}\n\n{section}"
    summary_path.write_text(summary_markdown, encoding="utf-8", newline="\n")
    return details_path, summary_path


def main() -> None:
    """
    Запускает анализ рецептов из командной строки.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Печатает пути к сохранённым таблицам.

    Fallbacks:
        Ошибки чтения, конфигурации и токенизации передаются вызывающей стороне.
    """
    # Выполняем анализ с параметрами CLI и показываем созданные файлы.
    args = parse_args()
    details_path, summary_path = analyze_recipes(
        recipes_path=args.recipes_path,
        output_dir=args.output_dir,
        experiment=args.experiment,
        settings_path=args.settings,
        sbert_tokenizer_name=args.sbert_tokenizer,
        qwen_tokenizer_name=args.qwen_tokenizer,
    )
    print(f"Детальная таблица: {details_path}")
    print(f"Сводная таблица: {summary_path}")


if __name__ == "__main__":
    main()
