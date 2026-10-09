import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "time_seconds": ("Время HTTP-запроса", "сек"),
    "completion_tokens": ("Выходные токены", "токены"),
    "prompt_tokens": ("Входные токены", "токены"),
    "tokens_per_second": ("Скорость по полному времени", "токенов/с"),
}


def normalize_mode(value):
    text = str(value).lower()
    if text.startswith("a") or "baseline" in text:
        return "A (базовый)"
    if text.startswith("b") or "tuned" in text:
        return "B (тюнинг)"
    return str(value)


def load_results(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Предупреждение: строка {lineno} пропущена: {e}")
                continue
            if not isinstance(item, dict):
                continue
            rows.append(item)
    if not rows:
        raise ValueError("В JSONL не найдено записей")
    df = pd.DataFrame(rows)
    for column in ["model", "task", "mode"]:
        if column not in df:
            raise ValueError(f"Отсутствует поле {column!r}")
    df["mode_label"] = df["mode"].map(normalize_mode)
    df["answer"] = (
        df.get("answer", pd.Series(index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
    )
    for metric in METRICS:
        if metric not in df:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if "tokens_per_second" in df and df["tokens_per_second"].isna().all():
        df["tokens_per_second"] = df["completion_tokens"] / df["time_seconds"].replace(
            0, np.nan
        )
    if "error" not in df:
        df["error"] = None
    df["success"] = df["error"].isna() & (df["answer"].str.len() > 0)
    df["answer_chars"] = df["answer"].str.len()
    df["has_answer"] = df["answer"].str.len() > 0
    return df


def plot_metric(df, metric, out):
    data = df[df["error"].isna()].dropna(subset=[metric]).copy()
    if data.empty:
        print(f"Пропуск графика {metric}: нет данных")
        return
    tasks = sorted(data["task"].unique())
    models = list(dict.fromkeys(data["model"]))
    modes = list(dict.fromkeys(sorted(data["mode_label"].unique())))
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(max(7, 5 * len(tasks)), 5), squeeze=False, sharey=True
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, task in zip(axes[0], tasks):
        subset = data[data["task"] == task]
        x = np.arange(len(models))
        width = 0.8 / len(modes)
        for j, mode in enumerate(modes):
            means, errors = [], []
            for model in models:
                values = subset[
                    (subset["model"] == model) & (subset["mode_label"] == mode)
                ][metric].dropna()
                means.append(values.mean() if len(values) else np.nan)
                errors.append(values.std(ddof=1) if len(values) > 1 else 0)
            ax.bar(
                x - 0.4 + width * (j + 0.5),
                means,
                width,
                yerr=errors,
                capsize=3,
                label=mode,
                color=colors[j % len(colors)],
            )
        ax.set_xticks(x, [m.split(":")[0] for m in models], rotation=20, ha="right")
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.2)
    axes[0][0].set_ylabel(METRICS[metric][1])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(modes)),
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(METRICS[metric][0], y=1.11, fontsize=14)
    fig.tight_layout()
    fig.savefig(out / f"{metric}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_repeatability(df, out):
    # Exact-match rate across repeats, by model/task/mode; not semantic stability.
    rows = []
    for key, group in df[df["success"]].groupby(["model", "task", "mode_label"]):
        answers = group["answer"].str.strip().tolist()
        if len(answers) < 2:
            continue
        n = len(answers)
        pairs = n * (n - 1) / 2
        matches = sum(
            answers[i] == answers[j] for i in range(n) for j in range(i + 1, n)
        )
        rows.append((*key, matches / pairs))
    if not rows:
        print("Пропуск графика повторяемости: необходимо минимум 2 успешных повтора")
        return None
    repeat = pd.DataFrame(
        rows, columns=["model", "task", "mode_label", "exact_match_rate"]
    )
    repeat.to_csv(out / "repeatability.csv", index=False)
    tasks = sorted(repeat["task"].unique())
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(max(7, len(tasks) * 5), 5), squeeze=False, sharey=True
    )
    modes = sorted(repeat["mode_label"].unique())
    models = list(dict.fromkeys(repeat["model"]))
    for ax, task in zip(axes[0], tasks):
        sub = repeat[repeat["task"] == task]
        x = np.arange(len(models))
        width = 0.8 / len(modes)
        for i, mode in enumerate(modes):
            vals = [
                sub.loc[
                    (sub.model == model) & (sub.mode_label == mode), "exact_match_rate"
                ].mean()
                for model in models
            ]
            ax.bar(x - 0.4 + width * (i + 0.5), vals, width, label=mode)
        ax.set_title(task)
        ax.set_xticks(x, [m.split(":")[0] for m in models], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
    axes[0][0].set_ylabel("Доля дословно совпавших пар")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(modes), bbox_to_anchor=(0.5, 1.03)
    )
    fig.suptitle("Повторяемость (строгое совпадение текста)", y=1.11)
    fig.tight_layout()
    fig.savefig(out / "repeatability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return repeat


def main():
    p = argparse.ArgumentParser(description="Графики для лабораторной LLM")
    p.add_argument(
        "--input", type=Path, default=Path(__file__).parent / "results/responses.jsonl"
    )
    p.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/plots"
    )
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f"Файл не найден: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    df = load_results(args.input)
    ok = df[df["error"].isna()]
    summary = (
        ok.groupby(["model", "task", "mode_label"], dropna=False)
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
    summary.to_csv(args.output / "summary.csv", index=False, encoding="utf-8-sig")
    failures = df[~df["error"].isna()]
    if not failures.empty:
        failures.to_csv(args.output / "errors.csv", index=False, encoding="utf-8-sig")
    for metric in METRICS:
        plot_metric(df, metric, args.output)
    plot_repeatability(df, args.output)
    print(f"Обработано записей: {len(df)}; ошибок: {len(failures)}")
    print(f"Результаты сохранены в: {args.output.resolve()}")
    print(
        "Примечание: скорость рассчитана по полному времени HTTP-запроса, включая накладные расходы."
    )
    print(
        "Качество, точность и галлюцинации требуют отдельной оценки; графики не измеряют их автоматически."
    )


if __name__ == "__main__":
    main()
