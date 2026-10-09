"""Запускает серию запросов к локальным моделям и сохраняет метрики."""

import csv
import json
import time
from pathlib import Path

from constants import (
    MODELS,
    MODES,
    OPENAI_BASE_URL,
    PROMPTS,
    REPEATS,
    RESULT_FIELDS,
)
from openai import OpenAI
from structures import ExperimentRecord, RequestMetrics


def request(
    client: OpenAI,
    model: str,
    prompt: str,
    params: dict[str, int | float],
) -> RequestMetrics:
    """
    Отправляет запрос модели и вычисляет метрики генерации.

    Parameters:
        client (OpenAI): Клиент OpenAI-совместимого API.
        model (str): Идентификатор модели.
        prompt (str): Текст пользовательского запроса.
        params (dict[str, int | float]): Параметры генерации.

    Returns:
        RequestMetrics: Ответ модели и измеренные метрики.

    Fallbacks:
        Недоступные значения usage сохраняются как None.
    """
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **params,
    )
    elapsed = time.perf_counter() - start
    usage = response.usage
    output_tokens = usage.completion_tokens if usage else None

    return RequestMetrics(
        answer=response.choices[0].message.content or "",
        time_seconds=round(elapsed, 4),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=output_tokens,
        tokens_per_second=(
            round(output_tokens / elapsed, 2)
            if output_tokens is not None and elapsed > 0
            else None
        ),
        finish_reason=response.choices[0].finish_reason,
    )


def main() -> None:
    """
    Выполняет эксперименты и сохраняет подробные и сводные результаты.

    Returns:
        None: Функция не возвращает значение.

    Fallbacks:
        Ошибка отдельного запроса записывается в результаты и не прерывает серию.
    """
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key="ollama",
        timeout=300.0,
        max_retries=0,
    )
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    installed = {model.id for model in client.models.list().data}
    missing = set(MODELS) - installed
    if missing:
        raise RuntimeError(f"Модели не загружены: {sorted(missing)}")

    records: list[ExperimentRecord] = []
    for model in MODELS:
        print(f"\nMODEL: {model}")

        # Прогреваем модель, чтобы первый запуск не искажал измерения.
        print("  Прогрев...")
        request(client, model, "Ответь одним словом: готов.", {})

        for task_name, prompt in PROMPTS.items():
            for mode_name, params in MODES.items():
                for repeat in range(1, REPEATS + 1):
                    print(f"  {task_name} | {mode_name} | repeat={repeat}")
                    try:
                        metrics = request(client, model, prompt, params)
                        record = ExperimentRecord(
                            model=model,
                            task=task_name,
                            mode=mode_name,
                            repeat=repeat,
                            prompt=prompt,
                            parameters=params.copy(),
                            answer=metrics.answer,
                            time_seconds=metrics.time_seconds,
                            prompt_tokens=metrics.prompt_tokens,
                            completion_tokens=metrics.completion_tokens,
                            tokens_per_second=metrics.tokens_per_second,
                            finish_reason=metrics.finish_reason,
                            error=None,
                        )
                        print("  Ответ:", metrics.answer[:150])
                    except Exception as error:
                        record = ExperimentRecord(
                            model=model,
                            task=task_name,
                            mode=mode_name,
                            repeat=repeat,
                            prompt=prompt,
                            parameters=params.copy(),
                            answer=None,
                            time_seconds=None,
                            prompt_tokens=None,
                            completion_tokens=None,
                            tokens_per_second=None,
                            finish_reason=None,
                            error=str(error),
                        )
                        print("  ОШИБКА:", error)

                    records.append(record)
                    with (output_dir / "responses.jsonl").open(
                        "a", encoding="utf-8"
                    ) as output_file:
                        output_file.write(
                            json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                        )

    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for record in records:
            record_data = record.to_dict()
            writer.writerow({key: record_data.get(key) for key in RESULT_FIELDS})

    print(f"\nГотово. Выполнено запросов: {len(records)}")
    print("Результаты:", output_dir.resolve())


if __name__ == "__main__":
    main()
