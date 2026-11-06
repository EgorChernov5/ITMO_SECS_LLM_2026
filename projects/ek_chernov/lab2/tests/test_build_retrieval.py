"""Проверяет выбор модели retrieval при построении индекса."""

from argparse import Namespace
from pathlib import Path

from source.scripts import build_index


def test_build_uses_only_linked_bi_encoder(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """
    Проверяет передачу Bi-Encoder в build без загрузки Cross-Encoder.

    Parameters:
        monkeypatch (object): Fixture подмены pytest.
        tmp_path (Path): Временный каталог pytest.

    Returns:
        None: Проверка завершается утверждениями pytest.

    Fallbacks:
        Фактическое построение эмбеддингов заменяется заглушкой.
    """
    settings = {
        "retrieval_models": {
            "sbert": {
                "type": "bi_encoder",
                "model": "sbert-model",
                "batch_size": 8,
            },
            "reranker": {
                "type": "cross_encoder",
                "model": "cross-model",
                "batch_size": 16,
            },
        },
        "retrievals": {
            "cross": {
                "type": "bi_n_cross",
                "bi_encoder": "sbert",
                "cross_encoder": "reranker",
                "candidate_top_k": 20,
                "top_k": 5,
            }
        },
        "experiments": {
            "cross": {
                "retrieval": "cross",
                "splitter": "markdown_header",
                "chunk_size": 160,
                "overlap": 20,
                "hnsw_m": 16,
                "ef_construct": 128,
                "ef_search": 64,
            }
        },
    }
    captured: list[tuple[object, ...]] = []
    args = Namespace(
        config="cross",
        settings=tmp_path / "pipeline.json",
        markdown=tmp_path / "recipes.md",
        batch_size=None,
        rebuild=False,
    )
    monkeypatch.setattr(build_index, "parse_args", lambda: args)
    monkeypatch.setattr(build_index, "load_pipeline_config", lambda path: settings)
    monkeypatch.setattr(
        build_index,
        "build_index",
        lambda *values: captured.append(values) or tmp_path,
    )
    build_index.main()
    assert captured[0][2:5] == ("sbert", "sbert-model", 8)
