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
    # Нормализуем пробелы и заменяем дробные Unicode-символы.
    content = [
        re.sub(
            r"\s+",
            " ",
            line.replace("¼", "1/4").replace("½", "1/2").replace("⁄", "/"),
        ).strip()
        for line in lines
    ]
    content = [line for line in content if line]

    # Восстанавливаем текст страниц, ошибочно распознанный как Markdown-таблицы.
    if 23 <= page_number <= 31:
        table_content: list[str] = []
        table_line = ""
        for line in content:
            if "|" not in line:
                if table_line:
                    table_content.append(table_line)
                    table_line = ""
                table_content.append(line)
                continue

            cells = [
                cell.strip()
                for cell in line.split("|")
                if cell.strip() and not re.fullmatch(r":?-{3,}:?", cell.strip())
            ]
            for cell in cells:
                starts_block = cell.startswith("●") or re.match(
                    r"^\d+(?:\.\d+)*[.)]\s*",
                    cell,
                )
                if starts_block and table_line:
                    table_content.append(table_line)
                    table_line = ""
                table_line = f"{table_line} {cell}".strip()
        if table_line:
            table_content.append(table_line)
        content = table_content

    # Сохраняем номер страницы только в служебном HTML-комментарии.
    blocks = [f"<!-- page: {page_number} -->"]

    if len(content) >= 2 and len(content[0]) == 1 and content[0].isupper():
        blocks.append(f"# {content.pop(0)}{content.pop(0)}")

    # Находим начала всех рецептов по повторяющемуся заголовку ингредиентов.
    ingredients_indexes = [
        index for index, line in enumerate(content) if line.lower() == "ингредиенты"
    ]
    if not ingredients_indexes:
        blocks.extend(content)
        return "\n\n".join(blocks)

    # Делим страницу перед названием каждого следующего рецепта.
    recipe_starts = [0, *(index - 1 for index in ingredients_indexes[1:]), len(content)]
    for start, end in zip(recipe_starts, recipe_starts[1:]):
        recipe = content[start:end]
        ingredients_index = next(
            index for index, line in enumerate(recipe) if line.lower() == "ингредиенты"
        )

        # Форматируем название и заголовок списка ингредиентов.
        title = " ".join(recipe[:ingredients_index])
        if title:
            blocks.append(f"## {title}")
        blocks.append("### ингредиенты")

        # Разделяем ингредиенты и способ приготовления текущего рецепта.
        method_index = next(
            (
                index
                for index, line in enumerate(recipe[ingredients_index + 1 :])
                if line.lower() == "способ приготовления"
            ),
            None,
        )
        ingredients = recipe[ingredients_index + 1 :]
        method: list[str] = []
        if method_index is not None:
            method = ingredients[method_index + 1 :]
            ingredients = ingredients[:method_index]

        # Собираем переносы строк внутри пунктов списка ингредиентов.
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

        # Форматируем нумерованные и маркированные шаги рецепта.
        if method_index is not None:
            blocks.append("### Способ приготовления")
            steps: list[str] = []
            for line in method:
                if line.startswith("●"):
                    steps.extend(
                        f"- {item.strip()}" for item in line.split("●") if item.strip()
                    )
                    continue
                step = re.match(r"^(\d+(?:\.\d+)*)[.)]\s*(?:\.\s*)?(.*)$", line)
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

    # Собираем документ и удаляем типографские артефакты исходного PDF.
    document = "\n\n".join(pages) + "\n"
    document = document.replace("—", "-").replace("❄", "")
    document = re.sub(r"[ \t]+,", ",", document)
    document = re.sub(r",(?=[^\s\d])", ", ", document)

    # Разделяем числовые значения, единицы измерения и сокращения.
    document = re.sub(
        r"(?<=\d)(?=(?:гр|г|шт|мл|ст|С)\b|(?:ст|ч)\.)",
        " ",
        document,
    )
    document = re.sub(r"\b(ст|ч)\.\s*л\.", r"\1. л.", document)
    document = re.sub(r"\b(ст|ч)\.\s*л\b", r"\1. л", document)
    document = re.sub(r"\b(гр|г|шт|мл)(?=\()", r"\1 ", document)
    document = re.sub(r"\b(гр|сах|л)\.(?=[А-Яа-яЁё])", r"\1. ", document)

    # Добавляем пробелы между словами, числами и открывающими скобками.
    document = re.sub(r"(?<=[А-Яа-яЁё])-(?=\d)", " - ", document)
    document = re.sub(r"(?<=[А-Яа-яЁё])(?=\()", " ", document)
    document = re.sub(r"\([ \t]+", "(", document)
    document = re.sub(r"[ \t]+\)", ")", document)
    document = re.sub(r"[ \t]{2,}", " ", document)
    document = re.sub(r"[ \t]+\n", "\n", document)

    # Сохраняем окончательно нормализованный Markdown.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        document,
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
