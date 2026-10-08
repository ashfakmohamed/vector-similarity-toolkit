import pytest

from vector_similarity_toolkit import cosine_similarity, dot_product, top_k


def test_similarity_and_ranking():
    assert dot_product([1, 2], [3, 4]) == 11
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert top_k([1, 0], {"x": [1, 0], "y": [0, 1]}, 1)[0][0] == "x"


def test_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1], [1, 2])
