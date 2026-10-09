"""Строит графики и сводные таблицы по результатам экспериментов."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import METRICS


def normalize_mode(value: object) -> str:
    """
    Преобразует техническое название режима в подпись для графика.

    Parameters:
        value (object): Исходное значение режима.

    Returns:
        str: Нормализованная подпись режима.

    Fallbacks:
        Неизвестное значение преобразуется в строку без изменения смысла.
    """
    text = str(value).lower()
    if text.startswith("a") or "baseline" in text:
        return "A (базовый)"
    if text.startswith("b") or "tuned" in text:
        return "B (тюнинг)"
    return str(value)


def load_results(path: Path) -> pd.DataFrame:
    """
    Загружает JSONL с результатами и подготавливает данные для анализа.

    Parameters:
        path (Path): Путь к файлу результатов.

    Returns:
        pd.DataFrame: Нормализованная таблица результатов.

    Fallbacks:
        Некорректные строки JSON пропускаются с предупреждением.
    """
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Предупреждение: строка {line_number} пропущена: {error}")
                continue
            if isinstance(item, dict):
                rows.append(item)

    if not rows:
        raise ValueError("В JSONL не найдено записей")

    data_frame = pd.DataFrame(rows)
    for column in ("model", "task", "mode"):
        if column not in data_frame:
            raise ValueError(f"Отсутствует поле {column!r}")

    data_frame["mode_label"] = data_frame["mode"].map(normalize_mode)
    data_frame["answer"] = (
        data_frame.get(
            "answer",
            pd.Series(index=data_frame.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
    )
    for metric in METRICS:
        if metric not in data_frame:
            data_frame[metric] = np.nan
        data_frame[metric] = pd.to_numeric(
            data_frame[metric],
            errors="coerce",
        )

    if data_frame["tokens_per_second"].isna().all():
        data_frame["tokens_per_second"] = data_frame["completion_tokens"] / data_frame[
            "time_seconds"
        ].replace(0, np.nan)

    if "error" not in data_frame:
        data_frame["error"] = None
    data_frame["success"] = data_frame["error"].isna() & (
        data_frame["answer"].str.len() > 0
    )
    data_frame["answer_chars"] = data_frame["answer"].str.len()
    data_frame["has_answer"] = data_frame["answer"].str.len() > 0
    return data_frame


def plot_metric(
    data_frame: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Строит столбчатые графики одной метрики по моделям и заданиям.

    Parameters:
        data_frame (pd.DataFrame): Таблица результатов.
        metric (str): Название столбца с метрикой.
        output_dir (Path): Каталог для сохранения графика.

    Returns:
        None: Функция не возвращает значение.

    Fallbacks:
        Если пригодных данных нет, график пропускается.
    """
    data = data_frame[data_frame["error"].isna()].dropna(subset=[metric]).copy()
    if data.empty:
        print(f"Пропуск графика {metric}: нет данных")
        return

    tasks = sorted(data["task"].unique())
    models = list(dict.fromkeys(data["model"]))
    modes = list(dict.fromkeys(sorted(data["mode_label"].unique())))
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(7, 5 * len(tasks)), 5),
        squeeze=False,
        sharey=True,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for axis, task in zip(axes[0], tasks):
        subset = data[data["task"] == task]
        positions = np.arange(len(models))
        width = 0.8 / len(modes)
        for mode_index, mode in enumerate(modes):
            means: list[float] = []
            errors: list[float] = []
            for model in models:
                values = subset[
                    (subset["model"] == model) & (subset["mode_label"] == mode)
                ][metric].dropna()
                means.append(values.mean() if len(values) else np.nan)
                errors.append(values.std(ddof=1) if len(values) > 1 else 0)
            axis.bar(
                positions - 0.4 + width * (mode_index + 0.5),
                means,
                width,
                yerr=errors,
                capsize=3,
                label=mode,
                color=colors[mode_index % len(colors)],
            )
        axis.set_xticks(
            positions,
            [model.split(":")[0] for model in models],
            rotation=20,
            ha="right",
        )
        axis.set_title(task)
        axis.grid(axis="y", alpha=0.2)

    axes[0][0].set_ylabel(METRICS[metric][1])
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(modes)),
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle(METRICS[metric][0], y=1.11, fontsize=14)
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{metric}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_repeatability(
    data_frame: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame | None:
    """
    Рассчитывает долю точных совпадений ответов и строит график.

    Parameters:
        data_frame (pd.DataFrame): Таблица результатов.
        output_dir (Path): Каталог для сохранения файлов.

    Returns:
        pd.DataFrame | None: Таблица повторяемости или None при нехватке данных.

    Fallbacks:
        Если успешных повторов меньше двух, файлы не создаются.
    """
    rows: list[tuple[str, str, str, float]] = []
    successful = data_frame[data_frame["success"]]
    for key, group in successful.groupby(["model", "task", "mode_label"]):
        answers = group["answer"].str.strip().tolist()
        if len(answers) < 2:
            continue

        pair_count = len(answers) * (len(answers) - 1) / 2
        matches = sum(
            answers[first] == answers[second]
            for first in range(len(answers))
            for second in range(first + 1, len(answers))
        )
        rows.append((*key, matches / pair_count))

    if not rows:
        print("Пропуск графика повторяемости: необходимо минимум 2 успешных повтора")
        return None

    repeatability = pd.DataFrame(
        rows,
        columns=[
            "model",
            "task",
            "mode_label",
            "exact_match_rate",
        ],
    )
    repeatability.to_csv(
        output_dir / "repeatability.csv",
        index=False,
    )
    tasks = sorted(repeatability["task"].unique())
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(7, len(tasks) * 5), 5),
        squeeze=False,
        sharey=True,
    )
    modes = sorted(repeatability["mode_label"].unique())
    models = list(dict.fromkeys(repeatability["model"]))

    for axis, task in zip(axes[0], tasks):
        subset = repeatability[repeatability["task"] == task]
        positions = np.arange(len(models))
        width = 0.8 / len(modes)
        for mode_index, mode in enumerate(modes):
            values = [
                subset.loc[
                    (subset["model"] == model) & (subset["mode_label"] == mode),
                    "exact_match_rate",
                ].mean()
                for model in models
            ]
            axis.bar(
                positions - 0.4 + width * (mode_index + 0.5),
                values,
                width,
                label=mode,
            )
        axis.set_title(task)
        axis.set_xticks(
            positions,
            [model.split(":")[0] for model in models],
            rotation=20,
            ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.2)

    axes[0][0].set_ylabel("Доля дословно совпавших пар")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(modes),
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle(
        "Повторяемость (строгое совпадение текста)",
        y=1.11,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "repeatability.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    return repeatability


def main() -> None:
    """
    Загружает результаты и сохраняет графики и сводные таблицы.

    Returns:
        None: Функция не возвращает значение.

    Fallbacks:
        Отсутствие входного файла обрабатывается стандартной ошибкой argparse.
    """
    parser = argparse.ArgumentParser(description="Графики для лабораторной LLM")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results/responses.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results/plots",
    )
    arguments = parser.parse_args()
    if not arguments.input.is_file():
        parser.error(f"Файл не найден: {arguments.input}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    data_frame = load_results(arguments.input)
    successful = data_frame[data_frame["error"].isna()]
    summary = (
        successful.groupby(
            ["model", "task", "mode_label"],
            dropna=False,
        )
        .agg(
            runs=("answer", "size"),
            mean_time_s=("time_seconds", "mean"),
            std_time_s=("time_seconds", "std"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            std_completion_tokens=("completion_tokens", "std"),
            mean_tokens_per_second=("tokens_per_second", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        arguments.output / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failures = data_frame[~data_frame["error"].isna()]
    if not failures.empty:
        failures.to_csv(
            arguments.output / "errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    for metric in METRICS:
        plot_metric(data_frame, metric, arguments.output)
    plot_repeatability(data_frame, arguments.output)

    print(f"Обработано записей: {len(data_frame)}; ошибок: {len(failures)}")
    print(f"Результаты сохранены в: {arguments.output.resolve()}")
    print(
        "Примечание: скорость рассчитана по полному времени HTTP-запроса, "
        "включая накладные расходы."
    )
    print(
        "Качество, точность и галлюцинации требуют отдельной оценки; "
        "графики не измеряют их автоматически."
    )


if __name__ == "__main__":
    main()
