"""Содержит контракты данных RAG-пайплайна."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """
    Хранит параметры извлечения текста из PDF.

    Attributes:
        use_ocr (bool): Признак принудительного OCR. По умолчанию: False.
        ocr_language (str): Язык Tesseract. По умолчанию: "rus".
        ocr_dpi (int): Разрешение OCR. По умолчанию: 300.

    Fallbacks:
        Автоматическое переключение между OCR и текстовым слоем не выполняется.
    """

    use_ocr: bool = False
    ocr_language: str = "rus"
    ocr_dpi: int = 300


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Хранит параметры чанкинга и векторного индекса.

    Attributes:
        name (str): Название конфигурации.
        splitter (str): Стратегия построения чанков и эмбеддингов.
        chunk_size (int): Максимальный размер чанка в токенах.
        overlap (int): Перекрытие соседних чанков в токенах.
        hnsw_m (int): Количество связей HNSW.
        ef_construct (int): Размер очереди при построении HNSW.
        ef_search (int): Размер очереди при поиске HNSW.

    Fallbacks:
        Не предусмотрены.
    """

    name: str
    splitter: str
    chunk_size: int
    overlap: int
    hnsw_m: int
    ef_construct: int
    ef_search: int


@dataclass(frozen=True, slots=True)
class TextUnit:
    """
    Хранит минимальную текстовую единицу для чанкинга.

    Attributes:
        text (str): Текст единицы.
        page (int): Номер исходной страницы.
        headings (tuple[str, ...]): Иерархия заголовков.

    Fallbacks:
        При отсутствии заголовков используется пустой кортеж.
    """

    text: str
    page: int
    headings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Описывает чанк и его метаданные.

    Attributes:
        id (str): Детерминированный идентификатор.
        text (str): Текст для поиска и генерации ответа.
        source (str): Название исходного документа.
        source_path (str): Относительный путь к документу.
        page_start (int): Первая страница чанка.
        page_end (int): Последняя страница чанка.
        headings (tuple[str, ...]): Иерархия заголовков.
        chunk_index (int): Порядковый номер чанка.
        content_hash (str): SHA-256 текста чанка.
        config_id (str): Идентификатор конфигурации.

    Fallbacks:
        Не предусмотрены.
    """

    id: str
    text: str
    source: str
    source_path: str
    page_start: int
    page_end: int
    headings: tuple[str, ...]
    chunk_index: int
    content_hash: str
    config_id: str


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """
    Описывает оценочный вопрос и эталонную разметку.

    Attributes:
        id (str): Идентификатор вопроса.
        question (str): Текст вопроса.
        relevant_pages (tuple[int, ...]): Номера релевантных страниц.
        relevant_chunks (tuple[str, ...]): Эталонные фрагменты документа.
        reference_facts (tuple[tuple[str, ...], ...]): Допустимые формулировки
            обязательных фактов ответа.

    Fallbacks:
        Не предусмотрены.
    """

    id: str
    question: str
    relevant_pages: tuple[int, ...]
    relevant_chunks: tuple[str, ...]
    reference_facts: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class RecipeMetrics:
    """
    Хранит рассчитанные параметры одного рецепта.

    Attributes:
        category (str): Категория рецепта.
        recipe (str): Название рецепта.
        sbert_tokens (int): Количество токенов SBERT.
        qwen_tokens (int): Количество токенов Qwen.
        chunk_count (int): Количество чанков после разбиения.
        character_count (int): Количество символов в тексте рецепта.
        word_count (int): Количество слов без чисел и пунктуации.

    Fallbacks:
        Не предусмотрены.
    """

    category: str
    recipe: str
    sbert_tokens: int
    qwen_tokens: int
    chunk_count: int
    character_count: int
    word_count: int


@dataclass(frozen=True, slots=True)
class GenerationPromptMetrics:
    """
    Хранит параметры generation-промпта одного оценочного вопроса.

    Attributes:
        question_id (str): Идентификатор оценочного вопроса.
        question (str): Текст оценочного вопроса.
        context_chunk_count (int): Число найденных чанков в контексте.
        instruction_tokens (int): Число токенов Qwen в инструкции.
        context_tokens (int): Число токенов Qwen в текстах контекста.
        question_tokens (int): Число токенов Qwen в вопросе.
        prompt_tokens (int): Число токенов Qwen в итоговом промпте.
        character_count (int): Число символов в итоговом промпте.
        word_count (int): Число слов без чисел и пунктуации.

    Fallbacks:
        При пустом результате retrieval контекст содержит ноль чанков.
    """

    question_id: str
    question: str
    context_chunk_count: int
    instruction_tokens: int
    context_tokens: int
    question_tokens: int
    prompt_tokens: int
    character_count: int
    word_count: int
