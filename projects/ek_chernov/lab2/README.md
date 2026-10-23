# Лабораторная работа №2 — RAG по сборнику рецептов

Проект преобразует PDF в Markdown, строит три dense-индекса, загружает их в Qdrant, сравнивает retrieval-метрики и отвечает через локальную модель Ollama.

## Схема запуска

Компоненты работают в двух окружениях:

- Windows-компьютер: Qdrant в Docker Compose;
- Ubuntu-контейнер на GPU-сервере: Python-пайплайн, SBERT и Ollama;
- доступ к Qdrant: обратный SSH-туннель.

```text
Ubuntu-контейнер:127.0.0.1:16333
              │
              │ обратный SSH-туннель
              ▼
Windows:127.0.0.1:6333 → Qdrant в Docker
```

Qdrant привязан только к localhost Windows-компьютера и напрямую в сеть не публикуется.

## Требования

На Windows нужны Docker Desktop, Docker Compose, OpenSSH Client и запись в `.ssh\config` для VPS.

В Ubuntu-контейнере нужны Python 3.12, NVIDIA GPU с CUDA, Ollama, а для OCR — Tesseract с языковыми данными `rus`.

Основные параметры находятся в `source/config/pipeline.json`. Для малого корпуса конфигурация принудительно строит HNSW: `indexing_threshold_kb=1` и `full_scan_threshold_kb=0`. Иначе Qdrant может использовать полный перебор, при котором сравнение `M`, `ef_construct` и `ef_search` некорректно.

## Первичная подготовка Ubuntu-контейнера

```bash
cd projects/ek_chernov/lab2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверка PyTorch и GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

Tesseract нужен только для запуска с `--ocr`:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus
tesseract --list-langs
```

В списке языков должен присутствовать `rus`.

## Полный запуск системы

Команды выполняются в указанном порядке.

### 1. Запуск Qdrant на Windows

Откройте PowerShell:

```powershell
cd .\projects\ek_chernov\lab2\

$env:QDRANT_BIND_ADDRESS = "127.0.0.1"
docker compose up -d qdrant
docker compose ps
curl.exe http://127.0.0.1:6333
```

Контейнер `ek-chernov-lab2-qdrant` должен перейти в состояние `running` или `healthy`.

### 2. Открытие обратного SSH-туннеля на Windows

Откройте отдельное окно PowerShell и оставьте его работающим:

```powershell
ssh -NT `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  -R 127.0.0.1:16333:127.0.0.1:6333 `
  Custodian
```

Если порт `16333` занят, выберите другой и используйте его далее в `QDRANT_URL`.

### 3. Подготовка Ubuntu-контейнера

```bash
cd ./projects/ek_chernov/lab2
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export QDRANT_URL=http://127.0.0.1:16333
export OLLAMA_URL=http://127.0.0.1:11434
```

Проверка Qdrant и GPU:

```bash
curl --connect-timeout 5 "$QDRANT_URL"
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

### 4. Запуск Ollama в Ubuntu-контейнере

Если Ollama ещё не запущена, выполните в отдельном терминале:

```bash
ollama serve
```

В основном терминале:

```bash
ollama pull qwen2.5:3b-instruct
ollama list
curl --connect-timeout 5 "$OLLAMA_URL/api/tags"
```

Код лабораторной не скачивает модель автоматически.

### 5. Конвертация PDF

Реализовано два скрипта по конвертации PDF в Markdown. Один основан на фреймворке PyMuPDF, второй на основе MarkItDown.

Для использования скрипта с PyMuPDF:

По существующему текстовому слою:

```bash
python -m source.scripts.prepare_markdown_pymupdf --no-ocr
```

С принудительным русскоязычным OCR:

```bash
python -m source.scripts.prepare_markdown_pymupdf --ocr
```

CLI-флаги имеют приоритет над `use_ocr` из конфигурации. Результат: `source/data/processed/recipes.md`.

Для использования скрипта с MarkItDown:

```bash
python -m source.scripts.prepare_markdown_markitdown
```

Результат: `source/data/processed/recipes_2.md`.

### 6. Построение трёх индексов

```bash
python -m source.scripts.build_index --config small --markdown "<path_to_md>" --rebuild
python -m source.scripts.build_index --config baseline --markdown "<path_to_md>" --rebuild
python -m source.scripts.build_index --config large --markdown "<path_to_md>" --rebuild
```

### 7. Загрузка индексов в Qdrant

Для воспроизводимого эксперимента пересоздайте коллекции. Скрипт дождётся состояния `green` и полной индексации HNSW; таймаут можно переопределить флагом `--index-timeout`:

```bash
python -m source.scripts.load_to_vector_store --config small --drop-and-reindex
python -m source.scripts.load_to_vector_store --config baseline --drop-and-reindex
python -m source.scripts.load_to_vector_store --config large --drop-and-reindex
```

Проверка коллекций:

```bash
curl "$QDRANT_URL/collections"
curl "$QDRANT_URL/collections/recipes_small"
curl "$QDRANT_URL/collections/recipes_baseline"
curl "$QDRANT_URL/collections/recipes_large"
```

В ответе каждой коллекции `indexed_vectors_count` должен быть равен `points_count`. Если параметры существующей коллекции отличаются от конфигурации, загрузчик потребует `--drop-and-reindex`.

### 8. Подготовка ручной разметки

Заполните `source/evaluation/questions.json` пятнадцатью вопросами после просмотра Markdown:

```json
[
  {
    "id": "q01",
    "question": "Вопрос по содержимому документа",
    "relevant_pages": [1]
  }
]
```

Номера страниц должны соответствовать маркерам `<!-- page: N -->`.

### 9. Оценка retrieval

```bash
python -m source.scripts.evaluate_retrieval
```

Для каждого вопроса выполняется один прогревочный и пять измеряемых запросов. Настройки находятся в секции `evaluation`; их можно переопределить флагами `--warmup-runs` и `--repetitions`. В отчёте используется медиана клиентского времени, включающего задержку SSH-туннеля. Все конфигурации проверяются через один туннель, поэтому условия сравнения одинаковы.

В `source/data/results` будут записаны:

- `retrieval_metrics.json`;
- `retrieval_report.md`;
- `best_config.json`.

### 10. RAG-запрос

```bash
python -m source.scripts.rag --question "Как приготовить выбранное блюдо?"
```

Запуск с явными параметрами:

```bash
python -m source.scripts.rag \
  --question "Как приготовить штрудель из лаваша?" \
  --config baseline \
  --top-k 15 \
  --ef-search 64
```

По умолчанию используется лучшая конфигурация из результатов оценки, а до оценки — `baseline`.

## Корректное завершение работы

Останавливайте компоненты в обратном порядке.

### 1. Завершение команд в Ubuntu

Дождитесь завершения индексирования или остановите активную команду сочетанием `Ctrl+C`.

Если `ollama serve` запускалась вручную, остановите её в соответствующем терминале сочетанием `Ctrl+C`. Не останавливайте общий Ollama-сервис, если им пользуются другие задачи.

Удалите переменные текущей shell-сессии и выйдите из окружения:

```bash
unset QDRANT_URL
unset QDRANT_API_KEY
unset OLLAMA_URL
unset CUDA_VISIBLE_DEVICES
deactivate
```

### 2. Закрытие SSH-туннеля на Windows

В окне PowerShell с `ssh -NT ... Custodian` нажмите `Ctrl+C`.

Проверка из Ubuntu:

```bash
nc -vz -w 3 127.0.0.1 16333
```

После закрытия туннеля подключение должно завершиться ошибкой.

### 3. Остановка Qdrant на Windows

```powershell
cd .\projects\ek_chernov\lab2

docker compose down
Remove-Item Env:QDRANT_BIND_ADDRESS -ErrorAction SilentlyContinue
```

`docker compose down` удаляет контейнер и служебную сеть, но сохраняет данные в `qdrant_storage`.

Если ранее создавалось временное правило Windows Firewall, удалите его:

```powershell
Get-NetFirewallRule `
  -DisplayName "Qdrant Lab2 TCP 6333" `
  -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule
```

Проверка закрытого порта:

```powershell
Get-NetTCPConnection `
  -LocalPort 6333 `
  -State Listen `
  -ErrorAction SilentlyContinue

Test-NetConnection 127.0.0.1 -Port 6333
```

Ожидаемый результат:

```text
TcpTestSucceeded : False
```

## Повторный запуск

При следующем запуске достаточно:

1. запустить Qdrant на Windows;
2. открыть обратный SSH-туннель;
3. активировать `.venv` и экспортировать переменные в Ubuntu;
4. запустить Ollama, если она не работает;
5. выполнять необходимые скрипты.

Markdown и локальные индексы пересоздавать необязательно, если PDF и параметры не изменились.

## Артефакты

В Git сохраняются исходный PDF, Markdown, конфигурация, ручная разметка и результаты. Чанки, эмбеддинги, данные Qdrant и кэши игнорируются.

## Тесты

Тесты не запускаются автоматически:

```bash
python -m pytest
```
