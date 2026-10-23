"""Преобразует исходный PDF в нормализованный Markdown."""

import argparse
import json
import re
from pathlib import Path
from statistics import median

from source.configuration import get_extraction_config, load_pipeline_config
from source.constants import CONFIG_PATH, MARKDOWN_PATH, PROJECT_DIR, RAW_PDF_PATH
from source.structures import ExtractionConfig


def parse_args() -> argparse.Namespace:
    """
    Разбирает параметры командной строки.

    Parameters:
        Не предусмотрены.

    Returns:
        argparse.Namespace: Аргументы запуска конвертации.

    Fallbacks:
        Пути и режим OCR берутся из конфигурации и констант проекта.
    """
    parser = argparse.ArgumentParser(
        description="Преобразование PDF с рецептами в Markdown."
    )
    parser.add_argument("--input", type=Path, default=RAW_PDF_PATH)
    parser.add_argument("--output", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument(
        "--ocr",
        action="store_true",
        dest="use_ocr",
        default=None,
        help="Принудительно распознать все страницы через Tesseract.",
    )
    ocr_group.add_argument(
        "--no-ocr",
        action="store_false",
        dest="use_ocr",
        help="Использовать существующий текстовый слой PDF.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    """
    Нормализует пробельные символы в извлечённой строке.

    Parameters:
        value (str): Исходная строка.

    Returns:
        str: Строка с одиночными пробелами.

    Fallbacks:
        Пустая строка остаётся пустой.
    """
    return re.sub(r"\s+", " ", value).strip()


def format_line(text: str, size: float, body_size: float, bold: bool) -> str:
    """
    Преобразует строку PDF в элемент Markdown.

    Parameters:
        text (str): Нормализованный текст строки.
        size (float): Максимальный размер шрифта строки.
        body_size (float): Типичный размер основного шрифта страницы.
        bold (bool): Признак полужирного начертания.

    Returns:
        str: Строка Markdown.

    Fallbacks:
        Неопознанная строка сохраняется как обычный текст.
    """
    bullet = re.match(r"^[•●▪◦]\s*(.+)$", text)
    if bullet:
        return f"- {bullet.group(1)}"
    if re.match(r"^\d+[.)]\s+", text):
        return re.sub(r"^(\d+)[)]", r"\1.", text)
    short = len(text) <= 120 and not text.endswith((".", ";", ",", ":"))
    if short and size >= body_size * 1.35:
        return f"## {text}"
    if short and (size >= body_size * 1.16 or bold):
        return f"### {text}"
    if short and len(text) > 3 and text.isupper():
        return f"## {text.title()}"
    return text


def page_to_markdown(page: object, extraction: ExtractionConfig) -> str:
    """
    Преобразует страницу PyMuPDF в Markdown.

    Parameters:
        page (object): Страница документа PyMuPDF.
        extraction (ExtractionConfig): Параметры извлечения и OCR.

    Returns:
        str: Нормализованное содержимое страницы.

    Fallbacks:
        При ошибке OCR возбуждает RuntimeError с инструкцией по Tesseract.
    """
    text_page = None
    if extraction.use_ocr:
        try:
            text_page = page.get_textpage_ocr(
                language=extraction.ocr_language,
                dpi=extraction.ocr_dpi,
                full=True,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "OCR не выполнен. Проверьте установку Tesseract и данных rus."
            ) from error

    page_data = page.get_text("dict", sort=True, textpage=text_page)
    blocks = page_data.get("blocks", [])
    sizes: list[float] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            sizes.extend(
                float(span.get("size", 0))
                for span in line.get("spans", [])
                if normalize_text(str(span.get("text", "")))
            )
    body_size = median(sizes) if sizes else 10.0

    paragraphs: list[str] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        block_lines: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = normalize_text(
                " ".join(str(span.get("text", "")) for span in spans)
            )
            if not text:
                continue
            size = max((float(span.get("size", body_size)) for span in spans))
            bold = any(int(span.get("flags", 0)) & 16 for span in spans)
            block_lines.append(format_line(text, size, body_size, bold))
        if block_lines:
            paragraphs.append("\n".join(block_lines))
    return "\n\n".join(paragraphs)


def convert_pdf(
    input_path: Path,
    output_path: Path,
    extraction: ExtractionConfig,
) -> None:
    """
    Конвертирует PDF в единый Markdown-документ.

    Parameters:
        input_path (Path): Путь к исходному PDF.
        output_path (Path): Путь к результирующему Markdown.
        extraction (ExtractionConfig): Параметры извлечения текста.

    Returns:
        None: Функция сохраняет документ на диск.

    Fallbacks:
        При отсутствии PDF возбуждает FileNotFoundError.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Исходный PDF не найден: {input_path}")

    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF не установлен. Установите зависимости из requirements.txt."
        ) from error

    relative_source = input_path.resolve().relative_to(PROJECT_DIR.resolve())
    with pymupdf.open(input_path) as document:
        header = [
            "---",
            f"title: {json.dumps(input_path.stem, ensure_ascii=False)}",
            f"source: {json.dumps(input_path.name, ensure_ascii=False)}",
            f"path: {json.dumps(relative_source.as_posix(), ensure_ascii=False)}",
            f"page_count: {document.page_count}",
            f"use_ocr: {str(extraction.use_ocr).lower()}",
            f"ocr_language: {extraction.ocr_language}",
            f"ocr_dpi: {extraction.ocr_dpi}",
            "---",
            "",
        ]
        pages: list[str] = []
        for page_number, page in enumerate(document, start=1):
            text = page_to_markdown(page, extraction)
            page_block = f"<!-- page: {page_number} -->"
            if text:
                page_block = f"{page_block}\n\n{text}"
            pages.append(page_block)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(header) + "\n\n".join(pages) + "\n"
    output_path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    """
    Запускает конвертацию PDF из командной строки.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Результат сохраняется в указанный файл.

    Fallbacks:
        CLI-флаг OCR имеет приоритет над значением конфигурации.
    """
    args = parse_args()
    extraction = get_extraction_config(load_pipeline_config(args.config))
    if args.use_ocr is not None:
        extraction = ExtractionConfig(
            use_ocr=args.use_ocr,
            ocr_language=extraction.ocr_language,
            ocr_dpi=extraction.ocr_dpi,
        )
    convert_pdf(args.input, args.output, extraction)
    print(f"Markdown сохранён: {args.output}")


if __name__ == "__main__":
    main()
