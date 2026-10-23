"""Преобразует PDF с рецептами в структурированный Markdown."""

import argparse
import re
from pathlib import Path

from source.constants import MARKDOWN_PATH, RAW_PDF_PATH


def normalize_page(page_number: int, lines: list[str]) -> str:
    """
    Форматирует содержимое одной страницы с рецептом.

    Parameters:
        page_number (int): Номер страницы PDF.
        lines (list[str]): Строки страницы, извлечённые MarkItDown.

    Returns:
        str: Страница в структурированном формате Markdown.

    Fallbacks:
        Неопознанные строки сохраняются как обычный текст.
    """
    content = [re.sub(r"\s+", " ", line).strip() for line in lines]
    content = [line for line in content if line]
    blocks = [f"<!-- page: {page_number} -->", str(page_number)]

    if len(content) >= 2 and len(content[0]) == 1 and content[0].isupper():
        blocks.append(f"# {content.pop(0)}{content.pop(0)}")

    ingredients_index = next(
        (index for index, line in enumerate(content) if line.lower() == "ингредиенты"),
        None,
    )
    if ingredients_index is None:
        blocks.extend(content)
        return "\n\n".join(blocks)

    title = " ".join(content[:ingredients_index])
    if title:
        blocks.append(f"## {title}")
    blocks.append("### ингредиенты")

    method_index = next(
        (
            index
            for index, line in enumerate(content[ingredients_index + 1 :])
            if line.lower() == "способ приготовления"
        ),
        None,
    )
    ingredients = content[ingredients_index + 1 :]
    method: list[str] = []
    if method_index is not None:
        method = ingredients[method_index + 1 :]
        ingredients = ingredients[:method_index]

    ingredient_blocks: list[str] = []
    bullets: list[str] = []
    for line in ingredients:
        if line.startswith("●"):
            bullets.append(f"- {line.removeprefix('●').strip()}")
        elif bullets and not line.endswith(":") and not line.isupper():
            bullets[-1] = f"{bullets[-1]} {line}"
        else:
            if bullets:
                ingredient_blocks.append("\n".join(bullets))
                bullets = []
            ingredient_blocks.append(line)
    if bullets:
        ingredient_blocks.append("\n".join(bullets))
    blocks.extend(ingredient_blocks)

    if method_index is not None:
        blocks.append("### Способ приготовления")
        steps: list[str] = []
        for line in method:
            step = re.match(r"^(\d+)[.)]\s*(?:\.\s*)?(.*)$", line)
            if step:
                steps.append(f"{step.group(1)}. {step.group(2)}")
            elif steps:
                steps[-1] = f"{steps[-1]} {line}"
            else:
                steps.append(line)
        blocks.append("\n".join(steps))

    return "\n\n".join(blocks)


def convert_pdf(input_path: Path, output_path: Path) -> None:
    """
    Конвертирует PDF через MarkItDown и нормализует результат.

    Parameters:
        input_path (Path): Путь к исходному PDF.
        output_path (Path): Путь к результирующему Markdown.

    Returns:
        None: Функция сохраняет документ на диск.

    Fallbacks:
        При отсутствии PDF возбуждает FileNotFoundError.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Исходный PDF не найден: {input_path}")

    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise RuntimeError(
            "MarkItDown не установлен. Установите зависимость markitdown[pdf]."
        ) from error

    raw_lines = MarkItDown().convert(input_path).text_content.splitlines()
    pages: list[str] = []
    page_lines: list[str] = []
    page_number = 2

    for raw_line in raw_lines:
        line = raw_line.strip()
        if line == str(page_number):
            if page_lines:
                pages.append(normalize_page(page_number - 1, page_lines))
            page_lines = []
            page_number += 1
        elif page_number > 2:
            page_lines.append(line)
    if page_lines:
        pages.append(normalize_page(page_number - 1, page_lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n\n".join(pages) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """
    Запускает конвертацию PDF из командной строки.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Результат сохраняется в указанный файл.

    Fallbacks:
        Пути к файлам берутся из констант проекта.
    """
    parser = argparse.ArgumentParser(
        description="Преобразование PDF с рецептами в Markdown через MarkItDown."
    )
    parser.add_argument("--input", type=Path, default=RAW_PDF_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=MARKDOWN_PATH.with_name("recipes_2.md"),
    )
    args = parser.parse_args()
    convert_pdf(args.input, args.output)
    print(f"Markdown сохранён: {args.output}")


if __name__ == "__main__":
    main()
