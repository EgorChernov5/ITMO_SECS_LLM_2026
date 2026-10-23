"""Проверяет преобразование страницы PDF в Markdown."""

from source.scripts.prepare_markdown import page_to_markdown
from source.structures import ExtractionConfig


class FakePage:
    """
    Имитирует страницу PyMuPDF.

    Attributes:
        ocr_args (dict[str, object] | None): Параметры вызванного OCR.
        text_page (object): Имитированный результат OCR.

    Fallbacks:
        Не предусмотрены.
    """

    def __init__(self) -> None:
        """
        Создаёт тестовую страницу.

        Parameters:
            Не предусмотрены.

        Returns:
            None: Инициализирует состояние страницы.

        Fallbacks:
            Не предусмотрены.
        """
        self.ocr_args: dict[str, object] | None = None
        self.text_page = object()

    def get_textpage_ocr(self, **kwargs: object) -> object:
        """
        Запоминает параметры OCR.

        Parameters:
            kwargs (object): Именованные параметры OCR.

        Returns:
            object: Имитированный TextPage.

        Fallbacks:
            Не предусмотрены.
        """
        self.ocr_args = kwargs
        return self.text_page

    def get_text(
        self,
        output: str,
        sort: bool,
        textpage: object | None,
    ) -> dict[str, object]:
        """
        Возвращает тестовый словарь текстовых блоков.

        Parameters:
            output (str): Формат результата.
            sort (bool): Признак сортировки.
            textpage (object | None): Результат OCR или None.

        Returns:
            dict[str, object]: Страница с заголовком и списком.

        Fallbacks:
            Не предусмотрены.
        """
        assert output == "dict"
        assert sort is True
        return {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "РЕЦЕПТ",
                                    "size": 20,
                                    "flags": 16,
                                }
                            ]
                        },
                        {
                            "spans": [
                                {
                                    "text": "• Ингредиент",
                                    "size": 10,
                                    "flags": 0,
                                }
                            ]
                        },
                    ],
                }
            ]
        }


def test_text_layer_does_not_start_ocr() -> None:
    """
    Проверяет обычное извлечение без OCR.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    page = FakePage()
    markdown = page_to_markdown(page, ExtractionConfig())
    assert page.ocr_args is None
    assert "## РЕЦЕПТ" in markdown
    assert "- Ингредиент" in markdown


def test_ocr_parameters() -> None:
    """
    Проверяет передачу языка и DPI в PyMuPDF OCR.

    Parameters:
        Не предусмотрены.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Не предусмотрены.
    """
    page = FakePage()
    page_to_markdown(page, ExtractionConfig(use_ocr=True))
    assert page.ocr_args == {"language": "rus", "dpi": 300, "full": True}
