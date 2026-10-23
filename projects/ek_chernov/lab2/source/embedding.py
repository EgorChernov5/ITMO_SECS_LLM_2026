"""Содержит GPU-эмбеддер для документов и поисковых запросов."""

class DenseEmbedder:
    """
    Строит нормализованные dense-векторы моделью SBERT.

    Attributes:
        model_name (str): Идентификатор модели Hugging Face.
        tokenizer (object): Токенизатор модели.
        model (object): Модель на CUDA.
        torch (object): Модуль PyTorch.

    Fallbacks:
        При отсутствии CUDA выполнение прекращается с RuntimeError.
    """

    def __init__(self, model_name: str) -> None:
        """
        Загружает токенизатор и модель на GPU.

        Parameters:
            model_name (str): Идентификатор модели Hugging Face.

        Returns:
            None: Создаёт готовый эмбеддер.

        Fallbacks:
            При отсутствии библиотек или CUDA возбуждает RuntimeError.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Для эмбеддингов установите зависимости из requirements.txt."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. Эмбеддер лабораторной должен работать на GPU."
            )
        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.model.eval()
        self.max_length = min(
            int(getattr(self.model.config, "max_position_embeddings", 512)),
            512,
        )

    def token_count(self, text: str) -> int:
        """
        Возвращает число токенов без специальных маркеров.

        Parameters:
            text (str): Исходный текст.

        Returns:
            int: Число токенов модели.

        Fallbacks:
            Пустой текст возвращает ноль.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_tokens(self, text: str) -> list[int]:
        """
        Кодирует текст без специальных токенов.

        Parameters:
            text (str): Исходный текст.

        Returns:
            list[int]: Идентификаторы токенов.

        Fallbacks:
            Пустой текст возвращает пустой список.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int]) -> str:
        """
        Декодирует токены в нормализованный текст.

        Parameters:
            tokens (list[int]): Идентификаторы токенов.

        Returns:
            str: Декодированный текст.

        Fallbacks:
            Пустой список возвращает пустую строку.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def embed(self, texts: list[str], batch_size: int = 8) -> object:
        """
        Строит L2-нормализованные эмбеддинги через mean pooling.

        Parameters:
            texts (list[str]): Тексты для векторизации.
            batch_size (int): Размер GPU-пакета. По умолчанию: 8.

        Returns:
            object: Матрица NumPy с типом float32.

        Fallbacks:
            Пустой вход возвращает матрицу размерности 0 × hidden_size.
        """
        import numpy as np

        if not texts:
            hidden_size = int(self.model.config.hidden_size)
            return np.empty((0, hidden_size), dtype=np.float32)

        vectors: list[object] = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (token_embeddings * mask).sum(1)
                pooled = pooled / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)
