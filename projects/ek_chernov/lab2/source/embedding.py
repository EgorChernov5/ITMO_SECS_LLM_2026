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
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", model_limit))
        self.max_length = min(
            model_limit,
            tokenizer_limit if tokenizer_limit < 1_000_000 else model_limit,
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

    def embed_token_batches(
        self,
        token_batches: list[list[int]],
    ) -> list[object]:
        """
        Возвращает token-level представления содержательных токенов пакета.

        Parameters:
            token_batches (list[list[int]]): Токены окон без специальных маркеров.

        Returns:
            list[object]: Матрицы NumPy без padding и специальных токенов.

        Fallbacks:
            Пустой пакет возвращает пустой список, слишком длинное окно отклоняется.
        """
        import numpy as np

        if not token_batches:
            return []
        # Добавляем special tokens и отмечаем исходные позиции независимо от token ID.
        prepared: list[dict[str, list[int]]] = []
        for tokens in token_batches:
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens)
            content_start = next(
                (
                    index
                    for index in range(len(input_ids) - len(tokens) + 1)
                    if input_ids[index : index + len(tokens)] == tokens
                ),
                None,
            )
            if content_start is None:
                raise ValueError(
                    "Не удалось сопоставить токены окна после добавления "
                    "специальных токенов."
                )
            prepared.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "content_mask": [0] * content_start
                    + [1] * len(tokens)
                    + [0] * (len(input_ids) - content_start - len(tokens)),
                    "token_type_ids": self.tokenizer.create_token_type_ids_from_sequences(
                        tokens
                    ),
                }
            )
        if any(len(item["input_ids"]) > self.max_length for item in prepared):
            raise ValueError(
                "Макроокно вместе со специальными токенами превышает контекст SBERT."
            )

        # Дополняем пакет вручную, чтобы fast-токенизатор не переоценивал [UNK].
        max_size = max(len(item["input_ids"]) for item in prepared)
        padding_side = self.tokenizer.padding_side
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Токенизатор SBERT не содержит pad_token_id.")
        for item in prepared:
            padding = max_size - len(item["input_ids"])
            if padding_side == "right":
                item["input_ids"].extend([pad_token_id] * padding)
                item["attention_mask"].extend([0] * padding)
                item["content_mask"].extend([0] * padding)
                item["token_type_ids"].extend([0] * padding)
            elif padding_side == "left":
                item["input_ids"] = [pad_token_id] * padding + item["input_ids"]
                item["attention_mask"] = [0] * padding + item["attention_mask"]
                item["content_mask"] = [0] * padding + item["content_mask"]
                item["token_type_ids"] = [0] * padding + item["token_type_ids"]
            else:
                raise ValueError("Поддерживается только левый или правый padding.")
        content_masks = self.torch.tensor(
            [item["content_mask"] for item in prepared],
            dtype=self.torch.bool,
            device="cuda",
        )
        encoded = {
            key: self.torch.tensor(
                [item[key] for item in prepared],
                dtype=self.torch.long,
                device="cuda",
            )
            for key in ("input_ids", "attention_mask", "token_type_ids")
        }
        with self.torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
        result: list[object] = []
        for index in range(len(token_batches)):
            result.append(
                hidden[index][content_masks[index]].cpu().numpy().astype(np.float32)
            )
        return result
