import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "qwen2.5:3b-instruct",
    "llama3.2:3b",
    "ministral-3:3b",
]


def api_request(path, data=None):
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data is not None else None

    return urllib.request.urlopen(
        urllib.request.Request(
            f"{OLLAMA_URL}{path}",
            data=body,
            headers=headers,
            method="POST" if data is not None else "GET",
        ),
        timeout=120,
    )


def installed_models():
    with api_request("/api/tags") as response:
        data = json.load(response)
    return {model["name"] for model in data.get("models", [])}


def download_model(model):
    print(f"\nЗагрузка модели: {model}")

    with api_request("/api/pull", {"model": model, "stream": True}) as response:
        for line in response:
            if not line.strip():
                continue

            data = json.loads(line)
            status = data.get("status", "")

            if "error" in data:
                raise RuntimeError(data["error"])

            total = data.get("total", 0)
            completed = data.get("completed", 0)

            if total:
                percent = completed / total * 100
                print(
                    f"\r{status}: {percent:.1f}%",
                    end="",
                    flush=True,
                )
            else:
                print(f"\n{status}")

    print(f"\nМодель {model} загружена")


def main():
    try:
        installed = installed_models()
    except urllib.error.URLError as e:
        print(f"Не удалось подключиться к Ollama: {e}")
        return

    print("Уже установленные модели:")
    for model in sorted(installed):
        print(f"  - {model}")

    for model in MODELS:
        if model in installed:
            print(f"\n{model} уже установлена. Пропускаем.")
            continue

        try:
            download_model(model)
        except Exception as e:
            print(f"\nОшибка загрузки {model}: {e}")

    print("\nПроверка итогового списка моделей:")
    final_models = installed_models()

    for model in MODELS:
        status = "OK" if model in final_models else "НЕ ЗАГРУЖЕНА"
        print(f"  {model}: {status}")


if __name__ == "__main__":
    main()
