import csv
import json
import time
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=300.0,
    max_retries=0,
)

MODELS = [
    "qwen2.5:3b-instruct",
    "llama3.2:3b",
    "ministral-3:3b",
]

PROMPTS = {
    "P1_generation": (
        "Напиши вежливое деловое письмо преподавателю "
        "с просьбой продлить срок сдачи лабораторной "
        "работы на два дня. Причина: технические "
        "проблемы при настройке окружения. "
        "Добавь тему, обращение, основной текст "
        "и подпись. Не более 150 слов."
    ),
    "P2_classification": (
        "Классифицируй обращение в одну из категорий: "
        "Billing, Tech support, Sales. "
        "Обращение: «После обновления приложения "
        "я не могу войти в аккаунт. Появляется "
        "ошибка авторизации». "
        "Ответь только названием категории."
    ),
    "P3_extraction": (
        "Извлеки информацию из описания товара. "
        "Верни только JSON с полями: name, price, "
        "color, memory_gb, manufacturer. "
        "Если значение отсутствует, используй null. "
        "Не придумывай данные. "
        "Описание: «Ноутбук Lenovo ThinkPad E14, "
        "16 ГБ оперативной памяти, корпус чёрного "
        "цвета, стоимость 74 990 рублей»."
    ),
}

MODES = {
    "A_baseline": {},
    "B_tuned": {
        "temperature": 0.2,
        "max_tokens": 256,
    },
}

# Минимум для лабораторной: 1
# Для анализа повторяемости рекомендуется 3
REPEATS = 3

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def request(model, prompt, params):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **params,
    )

    elapsed = time.perf_counter() - start
    usage = response.usage
    answer = response.choices[0].message.content or ""

    prompt_tokens = usage.prompt_tokens if usage else None
    output_tokens = usage.completion_tokens if usage else None

    return {
        "answer": answer,
        "time_seconds": round(elapsed, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "tokens_per_second": (
            round(output_tokens / elapsed, 2)
            if output_tokens is not None and elapsed > 0
            else None
        ),
        "finish_reason": response.choices[0].finish_reason,
    }


# Проверка наличия моделей
installed = {m.id for m in client.models.list().data}
missing = set(MODELS) - installed

if missing:
    raise RuntimeError(f"Модели не загружены: {sorted(missing)}")

records = []

for model in MODELS:
    print(f"\nMODEL: {model}")

    # Прогрев модели перед измерениями
    print("  Warmup...")
    request(model, "Ответь одним словом: готов.", {})

    for task_name, prompt in PROMPTS.items():
        for mode_name, params in MODES.items():
            for repeat in range(1, REPEATS + 1):
                print(f"  {task_name} | {mode_name} | repeat={repeat}")

                try:
                    metrics = request(model, prompt, params)

                    record = {
                        "model": model,
                        "task": task_name,
                        "mode": mode_name,
                        "repeat": repeat,
                        "prompt": prompt,
                        "parameters": params.copy(),
                        **metrics,
                        "error": None,
                    }

                    print("  Answer:", metrics["answer"][:150])

                except Exception as exc:
                    record = {
                        "model": model,
                        "task": task_name,
                        "mode": mode_name,
                        "repeat": repeat,
                        "prompt": prompt,
                        "parameters": params.copy(),
                        "answer": None,
                        "time_seconds": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "tokens_per_second": None,
                        "finish_reason": None,
                        "error": str(exc),
                    }
                    print("  ERROR:", exc)

                records.append(record)

                # Сохраняем результат после каждого запроса
                with (OUT / "responses.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Итоговая таблица
fields = [
    "model",
    "task",
    "mode",
    "repeat",
    "time_seconds",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
    "finish_reason",
    "error",
]

with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for record in records:
        writer.writerow({key: record.get(key) for key in fields})

print(f"\nГотово. Выполнено запросов: {len(records)}")
print("Результаты:", OUT.resolve())
