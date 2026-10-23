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
        chunk_size (int): Максимальный размер чанка в токенах.
        overlap (int): Перекрытие соседних чанков в токенах.
        hnsw_m (int): Количество связей HNSW.
        ef_construct (int): Размер очереди при построении HNSW.
        ef_search (int): Размер очереди при поиске HNSW.

    Fallbacks:
        Не предусмотрены.
    """

    name: str
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
    Описывает оценочный вопрос и эталонные страницы.

    Attributes:
        id (str): Идентификатор вопроса.
        question (str): Текст вопроса.
        relevant_pages (tuple[int, ...]): Номера релевантных страниц.

    Fallbacks:
        Не предусмотрены.
    """

    id: str
    question: str
    relevant_pages: tuple[int, ...]
